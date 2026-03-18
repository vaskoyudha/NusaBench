# NusaBench: Indonesian LLM Benchmark

NusaBench is a comprehensive benchmarking suite designed to evaluate the performance of Large Language Models (LLMs) on Indonesian language tasks. It provides a standardized framework for testing models across a variety of natural language processing (NLP) challenges, from sentiment analysis to machine translation and cultural knowledge.

## Why NusaBench?

As LLMs become more prevalent, understanding their capabilities in specific linguistic and cultural contexts is essential. Indonesian, with its unique morphological structure, informal variants, and rich cultural nuances, presents specific challenges that global benchmarks often overlook.

NusaBench aims to bridge this gap by providing:

*   **Diverse Task Selection**: 8 carefully curated tasks covering core NLP capabilities.
*   **Standardized Evaluation**: Consistent metrics and evaluation protocols for fair comparison.
*   **Multiple Backends**: Support for local models (HuggingFace), API-based models (Gemini), and benchmarking baselines.
*   **Ease of Use**: A powerful CLI and simple Python API for seamless integration into research and development workflows.

## Key Features

### Comprehensive Tasks
Evaluate models on sentiment analysis, natural language inference, question answering, named entity recognition, summarization, machine translation, toxicity detection, and Indonesian-specific cultural knowledge.

### Flexible Model Support
Use any model from the HuggingFace Hub, integrate with Google's Gemini API, or create custom model backends for specialized evaluation needs.

### Detailed Reporting
Get results in multiple formats, including rich console tables, structured JSON for automated analysis, and Markdown for documentation.

### Extensible Architecture
Designed with modularity in mind, making it easy to add new tasks, metrics, or model backends as the field evolves.

## Quick Glimpse

Evaluating a model is as simple as:

```bash
nusabench evaluate --model hf --model-args pretrained=indobenchmark/indobert-base-p1 --task sentiment_smsa --limit 100
```

Or via Python:

```python
import nusabench as nb

result = nb.evaluate(
    model="hf",
    model_args="pretrained=indobenchmark/indobert-base-p1",
    tasks=["sentiment_smsa"],
    limit=100
)
```

## Structure of this Documentation

*   **Getting Started**: Follow the installation guide and quick start tutorial to run your first evaluation.
*   **User Guide**: Detailed information on the CLI, Python API, available tasks, and supported model backends.
*   **Contributing**: Learn how to contribute to NusaBench, from bug fixes to new task implementations.
*   **Leaderboard**: Check the latest performance rankings of various models on the NusaBench suite.
