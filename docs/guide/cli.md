# CLI Reference

NusaBench provides a powerful and intuitive command-line interface (CLI) built with `typer`. This guide provides a detailed reference for all available commands and options.

## Base Command

The main entry point for the CLI is `nusabench`. You can see all available commands by running:

```bash
nusabench --help
```

## `evaluate`

The `evaluate` command is the core of NusaBench. It allows you to run evaluation for one or more tasks using a specified model backend.

### Usage

```bash
nusabench evaluate --model BACKEND [OPTIONS]
```

### Options

*   `--model TEXT`: **Required**. The name of the model backend to use. Currently supported backends:
    *   `dummy`: A baseline model that returns fixed or random labels (useful for testing pipelines).
    *   `hf`: HuggingFace model backend for local inference.
    *   `gemini`: Google Gemini API backend.
*   `--task TEXT`: **Required**. The name of the task to evaluate. This option can be provided multiple times to evaluate several tasks in a single run.
    *   Example: `--task sentiment_smsa --task nli_wrete`
*   `--model-args TEXT`: Comma-separated `key=value` strings for model configuration.
    *   For `hf`: `pretrained=model_name,device=cuda,dtype=float16`
    *   For `gemini`: `model=gemini-1.5-flash,temperature=0.0`
*   `--limit INTEGER`: Maximum number of samples to evaluate per task. Useful for quick tests or when working with large datasets.
*   `--output TEXT`: Path to save the evaluation results in JSON format. Defaults to `results.json`.
*   `--verbose / --no-verbose`: Enable or disable verbose logging during evaluation. Defaults to `--no-verbose`.

### Examples

**Evaluate a local HuggingFace model:**
```bash
nusabench evaluate \
    --model hf \
    --model-args pretrained=indobenchmark/indobert-base-p1 \
    --task sentiment_smsa \
    --limit 100 \
    --output indobert_results.json
```

**Evaluate multiple tasks with Gemini:**
```bash
export GEMINI_API_KEY=your_key
nusabench evaluate \
    --model gemini \
    --model-args model=gemini-1.5-pro \
    --task nli_wrete \
    --task qa_facqa \
    --limit 50
```

---

## `list-tasks`

Lists all evaluation tasks currently registered in NusaBench.

### Usage

```bash
nusabench list-tasks
```

This command will output a table containing the names of all 8 supported tasks, such as `sentiment_smsa`, `ner_nergrit`, and `cultural_indommu`.

---

## `list-models`

Lists all model backends currently registered in NusaBench.

### Usage

```bash
nusabench list-models
```

This command displays a table of available backends (`dummy`, `hf`, `gemini`). You can use any of these names with the `--model` option in the `evaluate` command.

---

## Global Options

*   `--help`: Show help message and exit.
*   `--version`: Show version and exit.
