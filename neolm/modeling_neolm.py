import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel
from transformers import initialization as init
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast, SequenceClassifierOutput
from transformers.cache_utils import Cache, DynamicCache

from .configuration_neolm import NeoLMConfig

# ─────────────────────────────────────────────
# RMSNorm
# ─────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


# ─────────────────────────────────────────────
# RoPE
# ─────────────────────────────────────────────

def precompute_rope_freqs(head_dim: int, seq_len: int, base: int = 10_000):
    assert head_dim % 2 == 0
    # Use float64 for accurate frequency computation, then store as float32
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).double() / head_dim))
    t        = torch.arange(seq_len).double()
    freqs    = torch.outer(t, inv_freq)
    return freqs.cos().float(), freqs.sin().float()

def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    offset: int = 0,
    base: int = 10_000,
):
    B, H, T, D = x.shape
    needed = offset + T

    if needed > cos.shape[0]:
        cos, sin = precompute_rope_freqs(
            head_dim=D,
            seq_len=needed + 64,
            base=base,
        )

    cos = cos[offset:offset + T].to(
        device=x.device, dtype=x.dtype
    )[None, None]

    sin = sin[offset:offset + T].to(
        device=x.device, dtype=x.dtype
    )[None, None]

    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    out = torch.empty_like(x)
    out[..., ::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos

    return out

# ─────────────────────────────────────────────
# Causal Self-Attention  (مع KV Cache)
# ─────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, config: NeoLMConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = (
                self.num_heads // self.num_key_value_heads
        )
        self.kv_dim = self.num_key_value_heads * self.head_dim

        self.qk_norm = config.qk_norm

        self.q_norm = (
            RMSNorm(self.head_dim, config.rms_norm_eps)
            if self.qk_norm else None
        )

        self.k_norm = (
            RMSNorm(self.head_dim, config.rms_norm_eps)
            if self.qk_norm else None
        )

        self.rope_base = config.rope_base
        self.dropout = config.dropout

        self.qkv = nn.Linear(config.hidden_size, config.hidden_size + 2 * self.kv_dim, bias=False)
        self.proj = nn.Linear(config.hidden_size,     config.hidden_size, bias=False)
        self.proj._residual_projection = config.scale_residual_init

        cos, sin = precompute_rope_freqs(
            self.head_dim, config.max_position_embeddings, config.rope_base
        )
        # persistent=False → مش بتتحفظ في الـ checkpoint، بتتحسب من جديد عند الـ init
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Cache | None = None,
        use_cache: bool = False,
        layer_idx: int = 0,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Cache | None]:

        B, T, C = x.shape

        q, k, v = self.qkv(x).split(
            [
                self.hidden_size,
                self.kv_dim,
                self.kv_dim,
            ],
            dim=-1,
        )

        def reshape(t, num_heads):
            return t.view(
                B, T, num_heads, self.head_dim
            ).transpose(1, 2)

        q = reshape(q, self.num_heads)
        k = reshape(k, self.num_key_value_heads)
        v = reshape(v, self.num_key_value_heads)

        # QK-Norm: normalize queries and keys per attention head
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # offset = عدد الـ tokens المحفوظة في الـ cache
        offset = 0
        if use_cache and past_key_value is not None:
            offset = past_key_value.get_seq_length(layer_idx)

        q = apply_rope(
            q,
            self.rope_cos,
            self.rope_sin,
            offset=offset,
            base=self.rope_base,
        )

        k = apply_rope(
            k,
            self.rope_cos,
            self.rope_sin,
            offset=offset,
            base=self.rope_base,
        )

        # KV Cache: update بيضيف الـ current k,v وبيرجع الكل
        if use_cache and past_key_value is not None:
            k, v = past_key_value.update(k, v, layer_idx)



        # Causal + Padding Mask
        key_length = k.size(-2)

        if attention_mask is not None:
            if (
                attention_mask.ndim != 2
                or attention_mask.shape[0] != B
                or attention_mask.shape[1] < key_length
            ):
                raise ValueError(
                    "Invalid attention_mask shape"
                )

            valid_keys = attention_mask[
                :, -key_length:
            ].to(device=q.device, dtype=torch.bool)

            q_pos = offset + torch.arange(
                T, device=q.device
            )
            k_pos = torch.arange(
                key_length, device=q.device
            )

            causal = k_pos[None, :] <= q_pos[:, None]

            attn_mask = (
                valid_keys[:, None, None, :]
                & causal[None, None, :, :]
            )
            is_causal = False

        elif offset == 0:
            attn_mask = None
            is_causal = True

        elif T == 1:
            attn_mask = None
            is_causal = False

        else:
            q_pos = offset + torch.arange(T, device=q.device)
            k_pos = torch.arange(key_length, device=q.device)

            attn_mask = k_pos[None, :] <= q_pos[:, None]
            is_causal = False

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
            enable_gqa=(
                    self.num_heads != self.num_key_value_heads
            ),
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y), past_key_value


# ─────────────────────────────────────────────
# SwiGLU MLP
# ─────────────────────────────────────────────

class SwiGLU(nn.Module):
    def __init__(self, config: NeoLMConfig):
        super().__init__()
        # intermediate_size محسوب في الـ config مرة واحدة
        hidden = config.intermediate_size

        self.w1   = nn.Linear(config.hidden_size, hidden, bias=False)
        self.w2   = nn.Linear(hidden, config.hidden_size, bias=False)
        self.w3   = nn.Linear(config.hidden_size, hidden, bias=False)
        self.w2._residual_projection = config.scale_residual_init
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ─────────────────────────────────────────────
# Transformer Block
# ─────────────────────────────────────────────

class NeoLMBlock(nn.Module):
    def __init__(self, config: NeoLMConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.ln1  = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln2  = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp  = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Cache | None = None,
        use_cache: bool = False,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Cache | None]:
        attn_out, past_key_value = self.attn(
            self.ln1(x),
            past_key_value=past_key_value,
            use_cache=use_cache,
            layer_idx=self.layer_idx,
            attention_mask=attention_mask,
        )
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, past_key_value


# ─────────────────────────────────────────────
# Base PreTrainedModel
# ─────────────────────────────────────────────

class NeoLMPreTrainedModel(PreTrainedModel):
    config_class                    = NeoLMConfig
    base_model_prefix               = "model"
    supports_gradient_checkpointing = False

    def _init_weights(self, module: nn.Module):

        if isinstance(module, nn.Linear):
            std = 0.02

            if (
                    self.config.scale_residual_init
                    and getattr(module, "_residual_projection", False)
            ):
                std /= math.sqrt(
                    2 * self.config.num_hidden_layers
                )

            init.normal_(
                module.weight,
                mean=0.0,
                std=std,
            )

            if module.bias is not None:
                init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

        elif isinstance(module, RMSNorm):
            init.ones_(module.weight)


# ─────────────────────────────────────────────
# NeoLMModel — Backbone
# ─────────────────────────────────────────────

class NeoLMModel(NeoLMPreTrainedModel):

    def __init__(self, config: NeoLMConfig):
        super().__init__(config)
        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks    = nn.ModuleList([
            NeoLMBlock(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.ln_f = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_init()

    def get_input_embeddings(self):
        return self.token_emb

    def set_input_embeddings(self, value):
        self.token_emb = value

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Cache | None]:

        B, T = input_ids.shape
        x = self.token_emb(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        for block in self.blocks:
            x, past_key_values = block(
                x,
                past_key_value=past_key_values,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )

        x = self.ln_f(x)
        return x, past_key_values


# ─────────────────────────────────────────────
# _TiedWeightsKeys — متوافق مع كل إصدارات HF
# ─────────────────────────────────────────────

class _TiedWeightsKeys(dict):
    """
    الإصدارات الجديدة تعمل .keys() → dict ✅
    الإصدارات القديمة تعمل iteration  → dict iteration = keys ✅
    """
    pass


# ─────────────────────────────────────────────
# NeoLMForCausalLM — الموديل الكامل
# ─────────────────────────────────────────────

class NeoLMForCausalLM(NeoLMPreTrainedModel, GenerationMixin):

    _tied_weights_keys              = _TiedWeightsKeys({"lm_head.weight": "model.token_emb.weight"})
    _keys_to_ignore_on_load_missing = [r"lm_head\.weight"]

    def __init__(self, config: NeoLMConfig):
        super().__init__(config)
        self.model   = NeoLMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        self._recompute_rope_buffers()

    # ── RoPE recompute (يمنع garbage values من القديم checkpoints) ──

    def _recompute_rope_buffers(self):
        device = next(self.parameters()).device
        for block in self.model.blocks:
            attn = block.attn
            cos, sin = precompute_rope_freqs(
                attn.head_dim,
                self.config.max_position_embeddings,
                self.config.rope_base,
            )
            attn.rope_cos = cos.to(device)
            attn.rope_sin = sin.to(device)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        model._recompute_rope_buffers()
        return model

    # ── Embeddings helpers ───────────────────────────────────────

    def get_input_embeddings(self):
        return self.model.token_emb

    def set_input_embeddings(self, value):
        self.model.token_emb = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def tie_weights(self, **kwargs):
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.token_emb.weight
        super().tie_weights(**kwargs)

    # ── Forward ──────────────────────────────────────────────────

    def forward(
        self,
        input_ids:       torch.Tensor,
        labels:          torch.Tensor | None = None,
        attention_mask:  torch.Tensor | None = None,   # للتوافق مع Trainer
        past_key_values: Cache | None        = None,
        use_cache:       bool | None         = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # لا نستخدم cache وقت التدريب
        if self.training:
            use_cache = False

        hidden, past_key_values = self.model(
            input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=(hidden,) if kwargs.get("output_hidden_states", False) else None
        )

    # ── Generate helpers ─────────────────────────────────────────

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> dict:
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0

        if past_len > 0:
            # في cache → نبعت آخر token بس
            input_ids = input_ids[:, -1:]
        else:
            # أول call → نقتص لو الـ prompt أطول من max_position_embeddings
            max_len = self.config.max_position_embeddings
            if input_ids.shape[1] > max_len:
                input_ids = input_ids[:, -max_len:]

        return {
            "input_ids":       input_ids,
            "past_key_values": past_key_values,
            "use_cache":       kwargs.get("use_cache", True),
            "attention_mask":  kwargs.get("attention_mask", None),
        }

class NeoLMForSequenceClassification(NeoLMPreTrainedModel):

    def __init__(self, config: NeoLMConfig):

        super().__init__(config)

        self.num_labels = config.num_labels

        # backbone
        self.model = NeoLMModel(config)

        # classifier head
        self.score = nn.Linear(
            config.hidden_size,
            config.num_labels,
            bias=True,
        )

        self.dropout = nn.Dropout(config.dropout)

        self.post_init()
        self._recompute_rope_buffers()

    def _recompute_rope_buffers(self):
        device = next(self.parameters()).device

        for block in self.model.blocks:
            attn = block.attn
            cos, sin = precompute_rope_freqs(
                attn.head_dim,
                self.config.max_position_embeddings,
                self.config.rope_base,
            )
            attn.rope_cos = cos.to(device)
            attn.rope_sin = sin.to(device)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        model = super().from_pretrained(
            pretrained_model_name_or_path, *args, **kwargs
        )
        model._recompute_rope_buffers()
        return model

    # ─────────────────────────────────────────
    # Embeddings Helpers
    # ─────────────────────────────────────────

    def get_input_embeddings(self):
        return self.model.token_emb

    def set_input_embeddings(self, value):
        self.model.token_emb = value

    # ─────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────

    def forward(

        self,

        input_ids: torch.Tensor,

        labels: torch.Tensor | None = None,

        attention_mask: torch.Tensor | None = None,

        **kwargs,

    ) -> SequenceClassifierOutput:

        # =====================================
        # Backbone
        # =====================================

        hidden, _ = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

        # =====================================
        # Pooling
        # =====================================

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            pooled = hidden.mean(dim=1)

        pooled = self.dropout(pooled)

        # =====================================
        # Classification Head
        # =====================================

        logits = self.score(pooled)

        # =====================================
        # Loss
        # =====================================

        loss = None

        if labels is not None:

            loss = F.cross_entropy(
                logits.view(-1, self.num_labels),
                labels.view(-1),
            )

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=(hidden,) if kwargs.get("output_hidden_states", False) else None,
        )