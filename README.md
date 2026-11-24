
# NeoLM

**A lightweight decoder-only Transformer built and pretrained from scratch.**

NeoLM is an open-source language modeling project focused on implementing, training, and evaluating small language models using PyTorch and Hugging Face Transformers.

The project covers the complete development pipeline, from corpus preparation and custom tokenizer training to model pretraining, inference, and benchmarking.

## NeoLM-Small-Base

The first released model contains **61.36 million parameters** and was pretrained on approximately **1.15 billion tokens** from NeoCorpus.

| Property | Value |
|---|---|
| Architecture | Dense Decoder-only Transformer |
| Parameters | 61,356,032 |
| Transformer Layers | 12 |
| Hidden Size | 512 |
| Attention | GQA (8 Query / 2 KV Heads) |
| Context Length | 1,024 |
| Tokenizer | Custom ByteLevel BPE (49,152 tokens) |
| Test Perplexity | 25.7605 |
| Test Bits per Byte | 0.9711 |

**Released resources:**

- [NeoLM-Small-Base — Model](https://huggingface.co/MaagDeveloper/NeoLM-Small-Base)
- [NeoCorpus — Dataset](https://huggingface.co/datasets/MaagDeveloper/NeoCorpus)

> **Note:** NeoLM-Small-Base is a base language model, not an instruction-tuned assistant. The release demonstrates an end-to-end training pipeline rather than state-of-the-art generation or reasoning capabilities.


## Architecture & Implementation

NeoLM implements a configurable, dense decoder-only Transformer using PyTorch, with integration into the Hugging Face Transformers ecosystem.

### Core Components

| Component | Implementation |
|---|---|
| Attention | Causal Grouped-Query Attention (GQA) |
| Positional Encoding | Rotary Position Embeddings (RoPE) |
| Normalization | RMSNorm with optional QK-Norm |
| Feed-Forward Network | SwiGLU |
| Attention Computation | PyTorch Scaled Dot-Product Attention (SDPA) |
| Residual Connections | Pre-Norm Transformer Blocks |
| Output Head | Linear projection with tied token embeddings |
| Inference | Autoregressive generation with KV Cache |

### Model Implementation

The architecture is implemented in two modules:

- [`configuration_neolm.py`](neolm/configuration_neolm.py) — Model configuration and architectural parameters.
- [`modeling_neolm.py`](neolm/modeling_neolm.py) — Transformer backbone, attention, feed-forward layers, and model heads.

The implementation includes:

- `NeoLMModel` — Transformer backbone.
- `NeoLMForCausalLM` — Autoregressive language modeling.
- `NeoLMForSequenceClassification` — Sequence classification.

Model configurations can be adjusted to experiment with different parameter counts, attention configurations, and context lengths.

The released **NeoLM-Small-Base** represents one specific configuration of this architecture.


## Training Pipeline

NeoLM provides an end-to-end training pipeline, from raw corpus preparation to model pretraining.

All stages are documented in Jupyter Notebooks and can be explored independently.

### 1. Corpus Preparation (Optional)

| Notebook | Description |
|---|---|
| [00_prepare_corpus](notebooks/corpus/00_prepare_corpus.ipynb) | Collect, filter, deduplicate, and prepare the foundation corpus. |
| [01_publish_corpus](notebooks/corpus/01_publish_corpus.ipynb) | Optionally publish the prepared dataset to Hugging Face. |

The corpus was derived from `karpathy/climbmix-400b-shuffle`, using the SmolLM-135M tokenizer for approximate token counting during collection.

The released dataset is available at [MaagDeveloper/NeoCorpus](https://huggingface.co/datasets/MaagDeveloper/NeoCorpus).

**Rebuilding the corpus is optional.** The training pipeline can use the published dataset directly.

### 2. Tokenizer & Pretraining

| Step | Notebook | Description |
|---|---|---|
| 1 | [Load Corpus](notebooks/training/00_load_corpus.ipynb) | Load NeoCorpus locally or from Hugging Face. |
| 2 | [Tokenizer Benchmark](notebooks/training/01_tokenizer_benchmark.ipynb) | Compare vocabulary sizes and tokenization efficiency. |
| 3 | [Train Tokenizer](notebooks/training/02_train_tokenizer.ipynb) | Train a 49,152-token ByteLevel BPE tokenizer from scratch. |
| 4 | [Prepare Token Streams](notebooks/training/03_prepare_token_streams.ipynb) | Convert documents into memory-mapped binary token streams. |
| 5 | [Pretrain NeoLM](notebooks/training/04_pretrain.ipynb) | Initialize and train NeoLM using Hugging Face Trainer. |

### 3. Training Data

The released NeoLM tokenizer produces the following token counts:

| Split | Documents | Tokens |
|---|---:|---:|
| Train | 1,889,952 | 1,147,218,931 |
| Validation | 5,000 | 3,084,132 |
| Test | 20,000 | 12,094,582 |

Token streams use `uint16` storage, with EOS separators between documents.

Training uses a context length of 1,024 and a stride of 1,023.

### 4. Training Features

The pretraining notebook supports:

- Random model initialization.
- BF16/FP16 mixed-precision training.
- Gradient accumulation.
- Cosine learning-rate scheduling and warmup.
- Periodic evaluation and perplexity logging.
- Checkpoint saving and training resumption.
- Early stopping and best-checkpoint selection.
- Final evaluation and Hugging Face model export.

### 5. Released Model & Reproducibility

The released **NeoLM-Small-Base** was pretrained from scratch. Its original training run was interrupted at approximately 202K optimizer steps and later extended through continued pretraining.

The cleaned training notebook provides a reusable implementation of the training pipeline. It is not presented as an exact replay of every historical training setting.

The released model weights are available on [Hugging Face](https://huggingface.co/MaagDeveloper/NeoLM-Small-Base).

Local datasets, token streams, and training checkpoints are excluded from Git.


## Inference

NeoLM-Small-Base supports autoregressive text generation with KV caching and configurable sampling strategies.

The [Inference Notebook](examples/inference.ipynb) demonstrates:

- Loading the released model and tokenizer.
- Text continuation with Temperature, Top-K, and Top-P sampling.
- Repetition penalty and streaming generation.
- EOS stopping diagnostics.
- Next-token probability analysis.

**Note:** NeoLM-Small-Base is a pretrained base model, not an instruction-tuned assistant.

## Evaluation & Benchmarks

The released model was evaluated on the NeoCorpus held-out test split and the HellaSwag validation benchmark.

### Language Modeling

| Metric | Result |
|---|---:|
| Test Cross-Entropy Loss | 3.2488 |
| Test Perplexity | 25.7605 |
| Bits per Byte (BPB) | 0.9711 |
| Evaluated Tokens | 12,094,581 |

The perplexity evaluation reproduces the previously reported result for NeoLM-Small-Base.

### HellaSwag — Zero-Shot

| Evaluation | Accuracy |
|---|---:|
| Random Baseline | 25.00% |
| Raw Accuracy | 28.03% |
| Length-Normalized Accuracy | 29.37% |

Evaluated on **10,042 validation examples** using a custom conditional log-likelihood scoring implementation.

These results are not directly comparable to official LM Evaluation Harness leaderboard scores without matching the reference evaluation protocol.

### Small Language Model Comparison

An exploratory comparison was conducted against similarly sized pretrained base language models.

| Model | Parameters | Raw Acc. | Norm. Acc. |
|---|---:|---:|---:|
| NeoLM-Small-Base | 61.4M | 31.50% | 28.00% |
| GPT2-Mini | 38.6M | 27.50% | 27.00% |
| Pythia-70M | 70.4M | 29.00% | 27.00% |
| DistilGPT2 | 81.9M | 29.50% | 31.00% |

The comparison uses the same **200 HellaSwag validation examples** and a custom scoring implementation.

Small differences should not be interpreted as statistically reliable rankings.

### Generation Quality

The generation benchmark explores different sampling temperatures and evaluates:

- English text fluency and readability.
- Story and dialogue continuation.
- Topic consistency and repetition.
- Factual and technical accuracy.
- EOS behavior and generation throughput.

NeoLM produces readable English continuations but exhibits limitations in factual reliability, reasoning, code generation, and long-range coherence.

### Evaluation Notebooks

| Notebook | Purpose |
|---|---|
| [Perplexity & BPB](benchmarks/01_perplexity.ipynb) | Full held-out test evaluation. |
| [Generation](benchmarks/02_generation.ipynb) | Text generation across sampling configurations. |
| [HellaSwag](benchmarks/03_hellaswag.ipynb) | Zero-shot commonsense continuation. |
| [Evaluation Summary](benchmarks/04_eval_summary.ipynb) | Consolidated NeoLM evaluation results. |
| [Model Comparison](benchmarks/05_model_comparison.ipynb) | Comparison with three small base models. |

Full benchmark outputs are available in [`benchmarks/results/`](benchmarks/results/).


## Installation & Quick Start

### Requirements

- Python 3.12
- PyTorch
- Hugging Face Transformers
- CUDA-compatible GPU (optional for inference)

### Installation

Clone the repository and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install -r requirements.txt
```

For GPU acceleration, ensure that your PyTorch installation supports your CUDA environment.

### Quick Inference

Load the released model directly from Hugging Face:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "MaagDeveloper/NeoLM-Small-Base"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
).to(device).eval()

prompt = "Once upon a time"

inputs = tokenizer(
    prompt,
    return_tensors="pt",
).to(device)

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.15,
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

`trust_remote_code=True` allows loading the custom NeoLM architecture. Review and trust the repository's modeling code before enabling it.

For additional sampling options, streaming, and token probability analysis, see [Inference Notebook](examples/inference.ipynb).

### Training from Scratch

The training pipeline is organized under [`notebooks/training/`](notebooks/training/).

Start with `00_load_corpus.ipynb` to download NeoCorpus from Hugging Face or use an existing local copy.

Then follow the notebooks in numerical order to train the tokenizer, prepare token streams, and pretrain the model.

Corpus reconstruction is optional and documented separately in [`notebooks/corpus/`](notebooks/corpus/).

**Note:** Full pretraining requires significantly more computation than inference. The repository also contains short training sanity checks for validating the pipeline before a full run.

### Local Artifacts

Generated datasets, token streams, model weights, and training checkpoints are stored under `artifacts/` and excluded from Git.

Released model weights and datasets are hosted on Hugging Face.

### Checkpoint Resumption

Training can resume from compatible Hugging Face Trainer checkpoints containing model, optimizer, scheduler, and training state.

With the current Transformers version, loading optimizer checkpoints requires PyTorch 2.6 or newer.

Model weights alone are not sufficient to reproduce an exact training resume.


## Acknowledgments

NeoLM builds on ideas, research, and tooling from the open-source machine learning community.

See [Acknowledgments & References](ACKNOWLEDGMENTS.md) for project credits and technical references.


## License

- **Source code and notebooks:** [MIT License](LICENSE).
- **NeoCorpus dataset:** [CC BY-NC 4.0](https://huggingface.co/datasets/MaagDeveloper/NeoCorpus).
- **Model weights:** See the [NeoLM-Small-Base Model Card](https://huggingface.co/MaagDeveloper/NeoLM-Small-Base) for licensing and usage terms.

NeoCorpus is derived from Nemotron-ClimbMix. Dataset users should review the original dataset's licensing conditions, including its non-commercial restrictions.

The MIT source-code license does not override third-party dataset or model-weight licenses.
