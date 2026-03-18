from __future__ import annotations

import logging
from typing import cast

from nusabench.metrics import MetricRegistry
from nusabench.models.base import Model
from nusabench.results import EvaluationResult, TaskResult
from nusabench.tasks.base import Task
from nusabench.utils.config import NusaBenchConfig

logger = logging.getLogger(__name__)
Document = dict[str, object]


def _split_generation_kwargs(
    generation_kwargs: dict[str, object],
) -> tuple[int, dict[str, object]]:
    max_tokens = generation_kwargs.get("max_tokens", 256)
    if not isinstance(max_tokens, int):
        max_tokens = 256
    extra_kwargs = {key: value for key, value in generation_kwargs.items() if key != "max_tokens"}
    return max_tokens, extra_kwargs


class Evaluator:
    def __init__(
        self,
        model: Model,
        tasks: list[Task],
        config: NusaBenchConfig | None = None,
    ) -> None:
        self.model: Model = model
        self.tasks: list[Task] = tasks
        self.config: NusaBenchConfig = config or NusaBenchConfig()

    def evaluate(self) -> EvaluationResult:
        results: dict[str, TaskResult] = {}
        for task in self.tasks:
            logger.info("Evaluating task: %s", task.config.task)
            dataset = [
                cast(Document, doc)
                for doc in task.load_dataset(split=task.config.test_split, limit=self.config.limit)
            ]
            processed = [cast(Document, task.preprocess_doc(doc)) for doc in dataset]
            predictions = self._run_task(task, processed)
            references = [task.format_target(doc) for doc in processed]

            # For multiple_choice tasks, normalize references and predictions to choice indices
            if task.config.output_type == "multiple_choice":
                choices = task.get_choices()
                if choices is not None:
                    # Normalize references to indices
                    normalized_refs: list[int | str] = []
                    for ref in references:
                        if ref.isdigit() or (ref.startswith("-") and ref[1:].isdigit()):
                            normalized_refs.append(int(ref))
                        elif ref in choices:
                            normalized_refs.append(choices.index(ref))
                        else:
                            normalized_refs.append(ref)
                    references = [str(r) for r in normalized_refs]

                    # Normalize predictions to indices
                    normalized_preds: list[int | str] = []
                    for pred in predictions:
                        if pred.isdigit() or (pred.startswith("-") and pred[1:].isdigit()):
                            normalized_preds.append(int(pred))
                        elif pred in choices:
                            normalized_preds.append(choices.index(pred))
                        else:
                            normalized_preds.append(pred)
                    predictions = [str(p) for p in normalized_preds]

            metrics = self._compute_metrics(task, predictions, references)
            results[task.config.task] = TaskResult(
                task_name=task.config.task,
                metrics=metrics,
                num_samples=len(dataset),
                model_name=self.model.model_name,
            )
        return EvaluationResult(results=results, model=self.model.model_name)

    def _run_task(self, task: Task, dataset: list[Document]) -> list[str]:
        prompts = [task.format_prompt(doc) for doc in dataset]
        output_type = task.config.output_type
        generation_kwargs = cast(dict[str, object], task.config.generation_kwargs)

        if output_type == "multiple_choice":
            return self._run_multiple_choice(task, prompts)
        if output_type == "loglikelihood" and self.model.supports_loglikelihood():
            targets = [task.format_target(doc) for doc in dataset]
            scores = self.model.loglikelihood(prompts, targets)
            return [str(score) for score in scores]

        max_tokens, extra_kwargs = _split_generation_kwargs(generation_kwargs)
        return self.model.generate(prompts, max_tokens=max_tokens, **extra_kwargs)

    def _run_multiple_choice(self, task: Task, prompts: list[str]) -> list[str]:
        choices = task.get_choices()
        if choices is None:
            message = (
                f"Task '{task.config.task}' has output_type='multiple_choice' but no "
                "target_choices defined in config"
            )
            raise ValueError(message)

        if self.model.supports_loglikelihood():
            predictions: list[str] = []
            for prompt in prompts:
                scores = [self.model.loglikelihood([prompt], [choice])[0] for choice in choices]
                best_idx = scores.index(max(scores))
                predictions.append(choices[best_idx])
            return predictions

        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        generation_kwargs = cast(dict[str, object], task.config.generation_kwargs)
        augmented_prompts: list[str] = []
        for prompt in prompts:
            choice_str = ", ".join(f"{letters[i]}) {choice}" for i, choice in enumerate(choices))
            augmented_prompts.append(f"{prompt}\nPilihan: {choice_str}\nJawaban:")

        max_tokens, extra_kwargs = _split_generation_kwargs(generation_kwargs)
        raw_answers = self.model.generate(augmented_prompts, max_tokens=max_tokens, **extra_kwargs)
        predictions = []
        for raw in raw_answers:
            normalized = raw.strip().upper()
            matched = False
            for i, letter in enumerate(letters[: len(choices)]):
                if normalized.startswith(letter):
                    predictions.append(choices[i])
                    matched = True
                    break

            if not matched:
                for choice in choices:
                    if choice.lower() in raw.lower():
                        predictions.append(choice)
                        matched = True
                        break

            if not matched:
                predictions.append(raw.strip())

        return predictions

    def _compute_metrics(
        self,
        task: Task,
        predictions: list[str],
        references: list[str],
    ) -> dict[str, float]:
        results: dict[str, float] = {}
        prediction_values = [cast(object, prediction) for prediction in predictions]
        reference_values = [cast(object, reference) for reference in references]
        for raw_metric_spec in task.config.metric_list:
            metric_spec = cast(dict[str, object], raw_metric_spec)
            metric_name = str(metric_spec["metric"])
            try:
                metric_cls = MetricRegistry.get(metric_name)
                metric = metric_cls()
                results.update(
                    metric.compute(predictions=prediction_values, references=reference_values)
                )
            except KeyError:
                logger.warning("Unknown metric '%s' — skipping", metric_name)
        return results
