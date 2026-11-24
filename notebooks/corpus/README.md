# NeoCorpus — Corpus Preparation

This directory documents how the foundation corpus used to train NeoLM was prepared and published.

The released dataset is available at [MaagDeveloper/NeoCorpus](https://huggingface.co/datasets/MaagDeveloper/NeoCorpus).

## Notebooks

| Notebook | Description |
|---|---|
| [00_prepare_corpus.ipynb](00_prepare_corpus.ipynb) | Collect, filter, deduplicate, and prepare the foundation corpus. |
| [01_publish_corpus (Optional).ipynb](01_publish_corpus%20(Optional).ipynb) | Optionally publish the prepared dataset to Hugging Face. |

## Corpus Preparation

The preparation pipeline includes:

1. Streaming documents from `karpathy/climbmix-400b-shuffle`.
2. Filtering documents by character length.
3. Approximate token counting with the `HuggingFaceTB/SmolLM-135M` tokenizer.
4. Resumable document collection.
5. Exact deduplication and dataset splitting.
6. Saving the prepared corpus locally.

The reference tokenizer is used only for estimating corpus size. NeoLM's final ByteLevel BPE tokenizer is trained separately from scratch.

## Dataset Statistics

| Split | Documents |
|---|---:|
| Train | 1,889,952 |
| Validation | 5,000 |
| Test | 20,000 |

The prepared local dataset is stored under `artifacts/pretrain_db/final/`, relative to the repository root.

Generated data files are excluded from Git.

## Using the Published Dataset

Rebuilding the corpus is optional.

To use the existing dataset directly, start with [00_load_corpus.ipynb](../training/00_load_corpus.ipynb).

The cleaned corpus preparation pipeline documents the processing workflow but is not claimed to reproduce the historical release byte-for-byte.

## Licensing

NeoCorpus is distributed under [CC BY-NC 4.0](https://huggingface.co/datasets/MaagDeveloper/NeoCorpus).

The corpus is derived from upstream ClimbMix data. Review the original dataset terms before redistribution or commercial use.

For the complete project, see the [main README](../../README.md).