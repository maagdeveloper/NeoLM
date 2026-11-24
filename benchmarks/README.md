# NeoLM — Evaluation & Benchmarks

Evaluation suite for [NeoLM-Small-Base](https://huggingface.co/MaagDeveloper/NeoLM-Small-Base), a 61.36M-parameter decoder-only Transformer pretrained from scratch.

The benchmarks measure language modeling performance, text generation behavior, and commonsense continuation.

## Benchmark Notebooks

| Notebook | Description |
|---|---|
| [01_perplexity.ipynb](01_perplexity.ipynb) | Test cross-entropy, perplexity, Bits per Byte, and inference performance |
| [02_generation.ipynb](02_generation.ipynb) | Text generation across prompts and sampling configurations |
| [03_hellaswag.ipynb](03_hellaswag.ipynb) | Zero-shot HellaSwag evaluation |
| [04_eval_summary.ipynb](04_eval_summary.ipynb) | Consolidated NeoLM evaluation results |
| [05_model_comparison.ipynb](05_model_comparison.ipynb) | Comparison with similarly sized pretrained base models |

## NeoLM-Small-Base Results

### Language Modeling

Evaluated on the held-out NeoCorpus test split using the released model.

| Metric | Result |
|---|---:|
| Test Tokens | 12,094,582 |
| Predicted Tokens | 12,094,581 |
| Test Loss | 3.2488 |
| Perplexity | 25.7605 |
| Bits per Byte | 0.9711 |

The evaluation reproduced the previously published perplexity result.

### HellaSwag

Evaluated on the full validation split (10,042 examples).

| Metric | Accuracy |
|---|---:|
| Random Baseline | 25.00% |
| Raw Accuracy | 28.03% |
| Length-Normalized Accuracy | 29.37% |

Results use a custom zero-shot conditional log-likelihood implementation, not an officially validated LM Evaluation Harness run.

### Small Model Comparison

Exploratory evaluation on the same 200 HellaSwag validation examples.

| Model | Parameters | Raw Accuracy | Normalized Accuracy |
|---|---:|---:|---:|
| NeoLM-Small-Base | 61.36M | 31.50% | 28.00% |
| GPT2-Mini | 38.60M | 27.50% | 27.00% |
| Pythia-70M | 70.43M | 29.00% | 27.00% |
| DistilGPT2 | 81.91M | 29.50% | 31.00% |

Small score differences should not be interpreted as statistically reliable rankings.

## Generation Evaluation

Each model was evaluated using 30 generation cases across three sampling configurations:

| Configuration | Temperature | Top-K | Top-P | Repetition Penalty |
|---|---:|---:|---:|---:|
| Conservative | 0.3 | 50 | 0.9 | 1.15 |
| Balanced | 0.7 | 50 | 0.9 | 1.15 |
| Creative | 0.8 | 50 | 0.9 | 1.15 |

Generation outputs are saved for qualitative inspection, including coherence, repetition, factual reliability, and topic consistency.

Generation throughput and EOS behavior are recorded but do not independently measure text quality.

## Reproducing the Evaluations

1. Install the dependencies from the repository's [`requirements.txt`](../requirements.txt).
2. Download the released model from [Hugging Face](https://huggingface.co/MaagDeveloper/NeoLM-Small-Base), or use the existing local model under `artifacts/model/`.
3. Run the relevant benchmark notebook.
4. Review the generated JSON files under [`results/`](results/).

The evaluation notebooks are independent of full model pretraining.

## Saved Results

```text
results/
├── neolm_perplexity.json
├── neolm_generation.json
├── neolm_hellaswag.json
└── comparison/
    ├── neolm_hellaswag.json
    ├── gpt2_mini_generation.json
    ├── gpt2_mini_hellaswag.json
    ├── pythia_70m_generation.json
    ├── pythia_70m_hellaswag.json
    ├── distilgpt2_generation.json
    └── distilgpt2_hellaswag.json
```

## Evaluation Limitations

- Perplexity is tokenizer-dependent. Bits per Byte is provided as an additional language-modeling metric.
- HellaSwag uses custom preprocessing and scoring; results may differ from official evaluation implementations.
- The 200-example model comparison is exploratory.
- Training datasets, training budgets, architectures, and tokenizers differ across models.
- Generation quality was reviewed qualitatively, without standardized human preference scoring.
- Throughput measurements may involve different hardware and numerical precisions, so they should not be treated as a controlled speed comparison.

These benchmarks document the capabilities and limitations of the released NeoLM-Small-Base model, rather than establishing the performance ceiling of the NeoLM architecture.

For project details, see the [main README](../README.md).