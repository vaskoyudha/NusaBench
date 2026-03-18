from __future__ import annotations

from pathlib import Path

from nusabench.tasks import Task, TaskRegistry, load_task_config


def test_mt_nusax_config_loads_and_registers(tmp_path: Path) -> None:
    cfg_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "nusabench"
        / "tasks"
        / "configs"
        / "mt_nusax.yaml"
    )
    config = load_task_config(cfg_path)

    assert config.task == "mt_nusax"
    assert config.output_type == "generate_until"
    assert any(m["metric"] == "bleu" for m in config.metric_list)
    assert any(m["metric"] == "chrf" for m in config.metric_list)
    assert config.target_choices is None
    assert config.generation_kwargs.get("max_tokens") == 256

    # Ensure templates render with expected fields (text_1=Indonesian, text_2=English)
    task = Task(config)
    sample_doc = {"text_1": "Halo dunia", "text_2": "Hello world"}
    prompt = task.format_prompt(sample_doc)
    target = task.format_target(sample_doc)
    assert "Halo dunia" in prompt
    assert target.strip() == "Hello world"

    # Register and retrieve via registry
    TaskRegistry.register(task)
    got = TaskRegistry.get("mt_nusax")
    assert got.config.task == "mt_nusax"
