# Model Backends

NusaBench supports multiple model backends, allowing you to evaluate everything from local HuggingFace models to commercial APIs like Google Gemini.

## Available Backends

### 1. `dummy` (Baseline)
The `dummy` backend is designed for testing and establishing a baseline. It doesn't perform actual LLM inference. Instead, it returns pre-configured or random responses based on the task type.

*   **When to use**: Testing evaluation pipelines, verifying metric calculations, or establishing a random/fixed baseline.
*   **Arguments**: None.

### 2. `hf` (HuggingFace)
The `hf` backend leverages the `transformers` library to run models locally. You can use any model available on the HuggingFace Hub or stored locally on your machine.

*   **When to use**: Evaluating open-source models (e.g., IndoBERT, GPT-2, Llama) with full control over hardware and inference settings.
*   **Key Arguments**:
    *   `pretrained` (Required): The HuggingFace model ID or local path.
    *   `device`: The device to run on (e.g., `"cuda"`, `"cpu"`, `"mps"`).
    *   `dtype`: The data type for model weights (e.g., `"float16"`, `"bfloat16"`, `"float32"`).
    *   `trust_remote_code`: Whether to allow custom code from the model repository.

**Example CLI usage:**
```bash
nusabench evaluate --model hf --model-args pretrained=indobenchmark/indobert-base-p1,device=cuda
```

### 3. `gemini` (Google Gemini API)
The `gemini` backend allows you to evaluate Google's latest generative models via their API.

*   **When to use**: Benchmarking state-of-the-art commercial models without needing local GPU resources.
*   **Key Arguments**:
    *   `model`: The specific Gemini model version (e.g., `"gemini-1.5-flash"`, `"gemini-1.5-pro"`). Defaults to `"gemini-1.5-flash"`.
    *   `temperature`: Sampling temperature. Defaults to `0.0` for reproducible results.
*   **Environment Variables**:
    *   `GEMINI_API_KEY`: Your Google AI Studio API key.

**Example CLI usage:**
```bash
export GEMINI_API_KEY=your_key
nusabench evaluate --model gemini --model-args model=gemini-1.5-pro
```

---

## Model Registry

All model backends are managed via the `ModelRegistry`. If you want to add a custom backend, you can inherit from the `Model` abstract base class and register your new class.

### Programmatic Access
You can list all registered models in Python:

```python
from nusabench.models import ModelRegistry

print(ModelRegistry.list())
# Output: ['dummy', 'hf', 'gemini']
```

### Custom Model Implementation
To add a new model backend, create a class that implements the `generate()` method:

```python
from nusabench.models.base import Model
from nusabench.models import ModelRegistry

@ModelRegistry.register("custom")
class MyCustomModel(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize your model here

    def generate(self, prompt: str) -> str:
        # Perform inference and return the generated text
        return "Your model's output"
```
