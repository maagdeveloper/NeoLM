# NeoLM — Training Pipeline

This directory contains the notebooks used to prepare and pretrain NeoLM from scratch.

The pipeline uses [NeoCorpus](https://huggingface.co/datasets/MaagDeveloper/NeoCorpus) and a custom ByteLevel BPE tokenizer.

## Notebooks

| Step | Notebook | Purpose |
|---|---|---|
| 0 | [00_load_corpus.ipynb](00_load_corpus.ipynb) | Load NeoCorpus locally or download it from Hugging Face. |
| 1 | [01_tokenizer_benchmark.ipynb](01_tokenizer_benchmark.ipynb) | Compare vocabulary sizes and tokenization efficiency. |
| 2 | [02_train_tokenizer.ipynb](02_train_tokenizer.ipynb) | Train the final 49,152-token ByteLevel BPE tokenizer. |
| 3 | [03_prepare_token_streams.ipynb](03_prepare_token_streams.ipynb) | Convert the dataset into binary token streams. |
| 4 | [04_pretrain.ipynb](04_pretrain.ipynb) | Initialize, train, evaluate, and save NeoLM. |

Run the notebooks in numerical order when reproducing the complete training pipeline.

## Data & Model Artifacts

Generated files are stored under `artifacts/` at the repository root.

```text
artifacts/
├── model/          # Tokenizer and saved model
├── pretrain_db/    # Dataset and binary token streams
└── checkpoints/    # Training checkpoints
```

These files are excluded from Git.

## Released Model Configuration

| Property | Value |
|---|---:|
| Parameters | 61,356,032 |
| Vocabulary | 49,152 |
| Transformer layers | 12 |
| Hidden size | 512 |
| Query / KV heads | 8 / 2 |
| Context length | 1,024 |

The architecture is implemented in [`neolm/`](../../neolm/).

## Training Notes

- The tokenizer benchmark is optional when the vocabulary size has already been selected.
- Full pretraining is computationally expensive; short sanity checks are included in the training notebook.
- Checkpoints can be used to resume compatible training runs.
- The current Transformers version requires PyTorch 2.6 or newer when loading optimizer checkpoints.
- The cleaned pipeline documents the training workflow but does not claim to reproduce every historical setting of the published model.

## Inference & Evaluation

To use the pretrained release without training, see:

- [Inference Notebook](../../examples/inference.ipynb)
- [Evaluation & Benchmarks](../../benchmarks/README.md)

The published model is available at [NeoLM-Small-Base](https://huggingface.co/MaagDeveloper/NeoLM-Small-Base).

For the full project overview, see the [main README](../../README.md).