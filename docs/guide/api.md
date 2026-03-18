# Python API Reference

NusaBench provides a simple and clean Python API for programmatic evaluation. This is ideal for researchers who want to integrate Indonesian LLM benchmarking into their existing scripts or automated experiment pipelines.

## Core Functions

The main functionality is exposed directly through the `nusabench` package.

### `evaluate()`

Run evaluation for one or more tasks with a given model.

```python
def evaluate(
    model: str,
    tasks: list[str],
    model_args: str | dict[str, str] | None = None,
    limit: int | None = None,
    verbose: bool = False,
) -> EvaluationResult:
```

**Parameters:**

*   **`model`** (`str`): The name of the model backend to use (e.g., `"hf"`, `"gemini"`, `"dummy"`).
*   **`tasks`** (`list[str]`): A list of task names to evaluate (e.g., `["sentiment_smsa", "nli_wrete"]`).
*   **`model_args`** (`str | dict[str, str] | None`): Arguments for the model constructor. Can be a comma-separated string (e.g., `"pretrained=gpt2"`) or a dictionary.
*   **`limit`** (`int | None`): Maximum number of samples to process per task. Defaults to `None` (all samples).
*   **`verbose`** (`bool`): If `True`, enables detailed logging. Defaults to `False`.

**Returns:**

*   `EvaluationResult`: An object containing the aggregated results for all tasks.

### `list_tasks()`

Returns a list of all registered task names.

```python
def list_tasks() -> list[str]:
```

---

## Result Objects

### `EvaluationResult`

The object returned by `evaluate()`. It contains a mapping of task names to their respective results.

*   **`results`** (`dict[str, TaskResult]`): A dictionary where keys are task names and values are `TaskResult` objects.

### `TaskResult`

Contains the results for a single task.

*   **`task_name`** (`str`): The name of the task.
*   **`metrics`** (`dict[str, float]`): A dictionary of calculated metrics (e.g., `{"accuracy": 0.85, "f1": 0.84}`).
*   **`samples`** (`list[dict]`): A list of processed samples, including model predictions and reference labels (only if verbose or configured).

---

## Usage Examples

### Basic Usage

```python
import nusabench as nb

# Evaluate using a HuggingFace model
results = nb.evaluate(
    model="hf",
    tasks=["sentiment_smsa"],
    model_args={"pretrained": "indobenchmark/indobert-base-p1"},
    limit=100
)

# Access specific metrics
smsa_metrics = results.results["sentiment_smsa"].metrics
print(f"Accuracy: {smsa_metrics['accuracy']:.4f}")
```

### Multiple Tasks and Custom Arguments

```python
import nusabench as nb

# Evaluate multiple tasks with specific Gemini settings
results = nb.evaluate(
    model="gemini",
    tasks=["nli_wrete", "qa_facqa"],
    model_args="model=gemini-1.5-flash,temperature=0.0",
    limit=50
)

for name, res in results.results.items():
    print(f"Task: {name}, Metrics: {res.metrics}")
```

### Listing Tasks Programmatically

```python
import nusabench as nb

available_tasks = nb.list_tasks()
print(f"Supported tasks: {', '.join(available_tasks)}")
```
