# NusaBench

**NLP Evaluation Suite for Bahasa Indonesia**

[![CI](https://github.com/vaskoyudha/NusaBench/actions/workflows/ci.yml/badge.svg)](https://github.com/vaskoyudha/NusaBench/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://vaskoyudha.github.io/NusaBench/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

NusaBench is a comprehensive evaluation framework for benchmarking language models
on Indonesian NLP tasks. It provides a unified interface for running evaluations
across 8 core tasks spanning sentiment analysis, NLI, question answering, NER,
summarization, machine translation, toxicity detection, and cultural knowledge.

## Features

- **8 evaluation tasks** covering core Indonesian NLP capabilities
- **3 model backends** — dummy (testing), HuggingFace Transformers, Google Gemini
- **CLI and Python API** for flexible integration
- **Static leaderboard** for tracking and comparing model performance
- **MkDocs documentation** with full API reference and guides
- **Automated metrics** including accuracy, F1, ROUGE, BLEU, chrF, and exact match
- **Configurable evaluation** with sample limits and verbose output
- **YAML-driven task configs** for easy extensibility

## Installation

```bash
pip install nusabench
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add nusabench
```

### Optional backends

```bash
# HuggingFace Transformers backend
pip install "nusabench[hf]"

# Google Gemini backend
pip install "nusabench[gemini]"
```

## Quick Start

### CLI

```bash
# List available tasks
nusabench list-tasks

# List available models
nusabench list-models

# Run evaluation with the dummy model
nusabench evaluate --model dummy --tasks sentiment_smsa --limit 10
```

### Python API

```python
import nusabench

# Run evaluation
result = nusabench.evaluate(
    model="dummy",
    tasks=["sentiment_smsa", "nli_wrete"],
    limit=10,
)

# Inspect results
for task_result in result.task_results:
    print(f"{task_result.task_name}: {task_result.metrics}")

# List all available tasks
print(nusabench.list_tasks())
```

## Tasks

NusaBench includes 8 evaluation tasks built on established Indonesian NLP datasets:

| Task | Dataset | HuggingFace ID | Metrics |
|------|---------|----------------|---------|
| `sentiment_smsa` | SmSA (IndoNLU) | `indonlp/indonlu` | accuracy, f1 |
| `nli_wrete` | WReTe (IndoNLU) | `indonlp/indonlu` | accuracy |
| `qa_facqa` | FacQA (IndoNLU) | `indonlp/indonlu` | exact_match |
| `ner_nergrit` | NERGrit (IndoNLU) | `indonlp/indonlu` | f1 |
| `summarization_indosum` | XL-Sum Indonesian | `csebuetnlp/xlsum` | rouge |
| `mt_nusax` | NusaX | `indonlp/nusa_x` | bleu, chrf |
| `toxicity_id` | CASA (IndoNLU) | `indonlp/indonlu` | accuracy, f1 |
| `cultural_indommu` | IndoMMLU | `indolem/IndoMMLU` | accuracy |

## Model Backends

| Backend | Install | Description |
|---------|---------|-------------|
| **dummy** | included | Returns fixed predictions; useful for testing and development |
| **hf** | `pip install "nusabench[hf]"` | HuggingFace Transformers models (local inference) |
| **gemini** | `pip install "nusabench[gemini]"` | Google Gemini API (requires `GEMINI_API_KEY`) |

## Leaderboard

Browse model performance across all tasks on the
[NusaBench Leaderboard](https://vaskoyudha.github.io/NusaBench/leaderboard/).

## Documentation

Full documentation is available at
[vaskoyudha.github.io/NusaBench](https://vaskoyudha.github.io/NusaBench/),
including:

- Getting started guide
- Task descriptions and dataset details
- API reference
- Contributing guidelines

## Contributing

Contributions are welcome! Please see the
[Contributing Guide](https://vaskoyudha.github.io/NusaBench/contributing/)
for guidelines on how to get started.

## License

NusaBench is licensed under the [Apache License 2.0](LICENSE).
