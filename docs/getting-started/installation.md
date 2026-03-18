# Installation Guide

NusaBench is designed to be easy to install and set up. Whether you're a researcher evaluating new models or a developer building Indonesian NLP applications, you can get started quickly with our Python-based suite.

## Prerequisites

Before installing NusaBench, ensure you have the following requirements met:

*   **Python 3.9 or higher**: NusaBench utilizes modern Python features for type safety and performance.
*   **pip**: The standard Python package installer.
*   **uv (Recommended)**: For faster dependency management and virtual environment handling.

## Standard Installation

The simplest way to install NusaBench is via pip from the repository.

```bash
# Clone the repository
git clone https://github.com/vaskoyudha/NusaBench.git
cd NusaBench

# Install using pip
pip install .
```

## Recommended Setup with `uv`

We highly recommend using `uv` for a fast and reliable installation experience.

```bash
# Clone the repository
git clone https://github.com/vaskoyudha/NusaBench.git
cd NusaBench

# Sync dependencies and create a virtual environment
uv sync
```

## Optional Dependencies

NusaBench supports multiple model backends, some of which require additional dependencies.

### Google Gemini API
To use the Gemini model backend, ensure you have an API key. No additional packages are required beyond the standard installation.

```bash
export GEMINI_API_KEY="your-api-key-here"
```

### Documentation
If you wish to build the documentation locally, install the `docs` extra:

```bash
uv sync --extra docs
```

## Verifying Installation

To verify that NusaBench is installed correctly, you can run the following command to list available tasks:

```bash
nusabench list-tasks
```

If successful, you should see a list of 8 supported Indonesian NLP tasks.

## Troubleshooting

### Dependency Conflicts
If you encounter dependency conflicts, we recommend using a clean virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install .
```

### Model-specific Issues
Some HuggingFace models may require specific libraries (e.g., `accelerate`, `bitsandbytes` for quantization). While NusaBench provides core support, these extra libraries should be installed manually if your specific model requires them.
