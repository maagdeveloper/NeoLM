# Acknowledgments & References

NeoLM was developed as an independent educational and engineering project focused on small-scale language model development.

The following open-source projects, research efforts, libraries, and datasets provided inspiration, technical references, and essential tooling.

## Research & Architectural Inspiration

- **[nanoGPT — Andrej Karpathy](https://github.com/karpathy/nanoGPT)** — Minimal GPT training implementations and practical language-model development.
- **[nanochat — Andrej Karpathy](https://github.com/karpathy/nanochat)** — End-to-end language-model training, evaluation, and research workflows.
- **[Llama — Meta](https://github.com/meta-llama/llama-models)** — Architectural reference for modern decoder-only Transformers.
- **[SmolLM2 — Hugging Face](https://huggingface.co/collections/HuggingFaceTB/smollm2)** — Research reference for efficient small language model pretraining and data-centric training.
- **[SmolLM3 — Hugging Face](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Base)** — Research reference for scaling small language models, training recipes, and open model development.


## Datasets & Tokenization

- **[Nemotron-ClimbMix — NVIDIA](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix)** — Original corpus family.
- **[ClimbMix-400B-Shuffle — Andrej Karpathy](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle)** — Source dataset used to construct NeoCorpus.
- **[SmolLM-135M — Hugging Face](https://huggingface.co/HuggingFaceTB/SmolLM-135M)** — Reference tokenizer for approximate corpus token counting.

The final NeoLM ByteLevel BPE tokenizer was trained separately from scratch.

## Libraries & Educational Resources

- **[PyTorch](https://pytorch.org/)** — Neural network implementation and training.
- **[Hugging Face Transformers](https://github.com/huggingface/transformers)** — Model integration, training, and generation.
- **[Hugging Face Tokenizers](https://github.com/huggingface/tokenizers)** — Tokenizer training and implementation.
- **[Hugging Face Datasets](https://github.com/huggingface/datasets)** — Dataset preparation and processing.
- **[Hugging Face Learn](https://huggingface.co/learn)** — Educational resources and documentation.
- **[LM Evaluation Harness — EleutherAI](https://github.com/EleutherAI/lm-evaluation-harness)** — Reference for language-model evaluation methodologies.

## Attribution & Licensing

Acknowledging a project does not imply that its source code or model weights were directly copied into NeoLM.

Third-party components, datasets, and model weights remain subject to their respective licenses and usage terms.

NeoLM's source-code license is documented separately in the repository's `LICENSE` file.