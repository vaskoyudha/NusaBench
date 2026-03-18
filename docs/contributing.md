# Contributing Guide

Thank you for your interest in contributing to NusaBench! We welcome all contributions, from bug reports and documentation improvements to new task implementations and model backend support.

## Code of Conduct

NusaBench is an inclusive project. We expect all contributors to adhere to standard respectful and collaborative practices.

## Setting Up Your Environment

To start contributing, follow these steps:

1.  **Fork and Clone**: Fork the repository on GitHub and clone it to your local machine.
2.  **Install Dependencies**: We recommend using `uv` for managing dependencies.
    ```bash
    uv sync --all-extras
    ```
3.  **Pre-commit Hooks**: Install pre-commit hooks to ensure consistent code style.
    ```bash
    uv run pre-commit install
    ```

## Development Workflow

1.  **Create a Branch**: Use a descriptive branch name for your changes (e.g., `feat/add-new-task` or `fix/metric-calculation`).
2.  **Write Code**: Implement your changes while following the existing code structure and conventions.
3.  **Run Tests**: Ensure that all existing tests pass and add new tests for your changes.
    ```bash
    uv run pytest tests/
    ```
4.  **Linting and Type Checking**: Run Ruff and MyPy to maintain code quality.
    ```bash
    uv run ruff check src/
    uv run mypy src/
    ```
5.  **Commit and Push**: Create atomic commits with clear messages.

## Adding a New Task

To add a new Indonesian NLP task to NusaBench:

1.  **YAML Configuration**: Create a new `.yaml` file in `src/nusabench/tasks/configs/`. Use existing tasks as a template.
2.  **Dataset**: Ensure the dataset is available on the HuggingFace Hub.
3.  **Metrics**: If the task requires a new metric, implement it in `src/nusabench/metrics/`.
4.  **Verification**: Run the task using the `dummy` model to verify the configuration and pipeline.

## Documentation

Improving documentation is one of the most impactful ways to contribute.

1.  **MkDocs**: Our documentation is built with MkDocs Material.
2.  **Build Docs**: To preview your changes locally:
    ```bash
    uv run mkdocs serve
    ```
3.  **Guidelines**: Write concise, accurate, and helpful technical prose. Avoid unnecessary filler or AI-generated clichés.

## Submitting Pull Requests

1.  **Summary**: Provide a clear summary of your changes in the pull request description.
2.  **Verification**: Ensure that all CI checks (tests, linting, build) pass.
3.  **Review**: Be open to feedback and suggestions from the maintainers.

---

Thank you for helping us build a better Indonesian LLM benchmarking suite!
