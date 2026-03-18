# Quick Start

Get up and running with NusaBench in minutes. This tutorial walks you through running your first Indonesian LLM evaluation using both the command-line interface and the Python API.

## 1. Verify Available Tasks

Start by checking the tasks supported by NusaBench:

```bash
nusabench list-tasks
```

You'll see a list of 8 Indonesian-specific NLP tasks, including sentiment analysis (`sentiment_smsa`), question answering (`qa_facqa`), and more.

## 2. Your First Evaluation (CLI)

The easiest way to run an evaluation is using the `evaluate` command. Let's start with a small test run using the `dummy` model backend on the sentiment analysis task:

```bash
nusabench evaluate --model dummy --task sentiment_smsa --limit 10
```

Wait, what if you want to test a real model? Use the `hf` (HuggingFace) backend:

```bash
nusabench evaluate \
    --model hf \
    --model-args pretrained=indobenchmark/indobert-base-p1 \
    --task sentiment_smsa \
    --limit 100
```

This command will:
1. Load the `indobert-base-p1` model from HuggingFace.
2. Run evaluation on the first 100 samples of the SMSA dataset.
3. Print a beautiful results table to your terminal.

## 3. Using the Python API

If you're integrating NusaBench into a larger pipeline, the Python API is the way to go:

```python
import nusabench as nb

# Simple evaluation with indobert
result = nb.evaluate(
    model="hf",
    model_args="pretrained=indobenchmark/indobert-base-p1",
    tasks=["sentiment_smsa", "nli_wrete"],
    limit=50,
)

# Print results
for task_name, task_result in result.results.items():
    print(f"--- {task_name} ---")
    print(f"Metrics: {task_result.metrics}")
```

## 4. Evaluating with Gemini

To evaluate API-based models like Gemini, set your API key first:

```bash
export GEMINI_API_KEY=your_key_here
nusabench evaluate \
    --model gemini \
    --model-args model=gemini-1.5-flash \
    --task cultural_indommu \
    --limit 20
```

## 5. Saving Results

You can export your evaluation results to a JSON file for further analysis:

```bash
nusabench evaluate \
    --model dummy \
    --task sentiment_smsa \
    --limit 50 \
    --output results.json
```

## Next Steps

Now that you've run your first evaluation, dive deeper into the documentation:

*   **CLI Reference**: Learn all available flags and commands.
*   **Python API**: Explore the full programmatic interface.
*   **Tasks Guide**: Understand the details of the 8 supported Indonesian tasks.
*   **Models Guide**: Learn about different backends and configuration options.
