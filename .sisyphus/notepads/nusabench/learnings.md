# NusaBench Learnings

## Project Setup
- Working directory: /home/vascosera/Documents/Github/NusaBench
- Git remote: https://github.com/vaskoyudha/NusaBench.git
- Branch: main
- Python version: 3.11
- Package manager: uv
- Layout: src-layout (src/nusabench/)

## Architecture Decisions
- Standalone framework — NOT a plugin to lm-eval-harness
- YAML-driven task configs (one file per task in src/nusabench/tasks/configs/)
- Two eval modes: loglikelihood (HF models) + generate_until (all models)
- multiple_choice: third mode — loglikelihood-per-choice (HF) or generate+parse (Gemini)
- All datasets loaded at runtime from HuggingFace Hub — NEVER bundled
- Zero-shot only (num_fewshot=0 always)

## Dataset Mapping (locked)
- Sentiment: indonlp/indonlu config=smsa
- NLI: indonlp/indonlu config=wrete
- QA: indonlp/indonlu config=facqa
- NER: indonlp/indonlu config=nergrit
- Summarization: indolem/indosum or csebuetnlp/xlsum config=indonesian
- MT: indonlp/nusax config=ind
- Toxicity: id_hatespeech or similar (TBD — agent must verify on HF Hub)
- Cultural: indolem/IndoMMLU (default config)

## Key Constraints
- Apache 2.0 license
- Each commit must pass: ruff check + mypy + existing tests
- No vLLM, OpenAI, Anthropic backends
- No dynamic leaderboard — static HTML+JSON only
- No Gradio/Streamlit
- Package name: nusabench, version: 0.1.0

- Added core model abstraction under src/nusabench/models with registry-backed DummyModel for CI-safe testing.
- Avoided direct bottom-of-module runtime import in models/__init__.py by using importlib preload plus __getattr__, which satisfies registration and Ruff E402.
- DummyModel loglikelihood returns one float per prompt entry to support batched prompt/target evaluation.

## [2026-03-19] Task 5: Utility Modules


## [2026-03-19] Task 6: CI Workflow
- CI uses uv sync --extra dev (not --all-extras) to avoid torch timeout
- Python matrix versions quoted as strings to prevent 3.10→3.1 YAML float parsing

## [2026-03-19] Task 4: Metrics Module
- Metric registry auto-registration can be done safely from metrics/__init__.py after importing concrete classes, avoiding per-module registry imports and import cycles.
- evaluate accuracy/f1 wrappers need local cache_dir under ~/.cache/nusabench/evaluate and benefit from cached loader helpers.
- sacrebleu corpus_bleu should use references shaped as a single list of references per corpus and use_effective_order=True to keep exact-match single-sentence BLEU high.
- basedpyright is not preinstalled in the environment; uv tool install basedpyright restores lsp_diagnostics verification.

## [2026-03-19] Task 7: Evaluator Engine
- Evaluator keeps execution sequential: load dataset, preprocess docs, route by output_type, then aggregate metrics into EvaluationResult.
- Convenience evaluate() should lazy-load registries to avoid circular import problems while preserving package-level ergonomics.
- Targeted evaluator tests should patch Task.load_dataset to avoid real HuggingFace downloads and keep CI deterministic.

## [2026-03-19] Task 8: HuggingFace Backend
- Optional HuggingFace backends can be registered safely from models/__init__.py via importlib.import_module so the decorator side effect runs without making the package hard-fail when extras are absent.
- Unit tests for optional ML backends stay CI-safe by patching sys.modules/import paths and mocking tokenizer/model behavior; real model downloads should remain behind @pytest.mark.slow.
