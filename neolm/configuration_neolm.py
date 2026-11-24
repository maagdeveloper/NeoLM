import math
from transformers import PretrainedConfig

class NeoLMConfig(PretrainedConfig):
    model_type = "neolm"

    def __init__(
            self,
            vocab_size: int = 32000,
            max_position_embeddings: int = 512,
            hidden_size: int = 512,
            num_attention_heads: int = 8,
            num_key_value_heads: int | None = None,
            num_hidden_layers: int = 8,
            intermediate_size: int | None = None,
            dropout: float = 0.0,
            tie_word_embeddings: bool = True,
            rope_base: int = 10_000,
            mlp_multiple_of: int = 256,
            rms_norm_eps: float = 1e-6,
            qk_norm: bool = True,
            scale_residual_init: bool = True,
            use_cache: bool = False,
            pad_token_id: int = None,
            bos_token_id: int = None,
            eos_token_id: int = None,
            **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

        if num_attention_heads <= 0 or hidden_size % num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads"
            )

        if (hidden_size // num_attention_heads) % 2 != 0:
            raise ValueError(
                "RoPE requires an even attention head dimension"
            )

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        if not (
                1 <= num_key_value_heads <= num_attention_heads
                and num_attention_heads % num_key_value_heads == 0
        ):
            raise ValueError(
                "num_key_value_heads must be positive "
                "and divide num_attention_heads"
            )

        if intermediate_size is None:
            intermediate_size = mlp_multiple_of * math.ceil(
                int(8 * hidden_size / 3) / mlp_multiple_of
            )

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_base = rope_base
        self.mlp_multiple_of = mlp_multiple_of
        self.rms_norm_eps = rms_norm_eps
        self.qk_norm = qk_norm
        self.scale_residual_init = scale_residual_init
        self.use_cache = use_cache
