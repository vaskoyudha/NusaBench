from __future__ import annotations

from pathlib import Path

from nusabench.tasks import Task
from nusabench.tasks.loader import load_task_config
from nusabench.tasks.ner_utils import format_entities, join_tokens, parse_generated_entities


def test_join_tokens_basic() -> None:
    tokens = ["Joko", "tinggal", "di", "Jakarta"]
    result = join_tokens(tokens)
    assert result == "Joko tinggal di Jakarta"


def test_join_tokens_empty() -> None:
    result = join_tokens([])
    assert result == ""


def test_join_tokens_single() -> None:
    result = join_tokens(["Joko"])
    assert result == "Joko"


def test_format_entities_basic() -> None:
    tokens = ["Joko", "tinggal", "di", "Jakarta"]
    tags = ["B-PER", "O", "O", "B-LOC"]
    result = format_entities(tokens, tags)
    assert "PER: Joko" in result
    assert "LOC: Jakarta" in result
    assert result == "PER: Joko, LOC: Jakarta"


def test_format_entities_multiword() -> None:
    tokens = ["PT", "Mitra", "Utama", "di", "Surabaya"]
    tags = ["B-ORG", "I-ORG", "I-ORG", "O", "B-LOC"]
    result = format_entities(tokens, tags)
    assert "ORG: PT Mitra Utama" in result
    assert "LOC: Surabaya" in result


def test_format_entities_consecutive() -> None:
    tokens = ["Joko", "bekerja", "di", "Jakarta", "dan", "Bandung"]
    tags = ["B-PER", "O", "O", "B-LOC", "O", "B-LOC"]
    result = format_entities(tokens, tags)
    assert "PER: Joko" in result
    assert "LOC: Jakarta" in result
    assert "LOC: Bandung" in result


def test_format_entities_no_entities() -> None:
    tokens = ["Joko", "tinggal"]
    tags = ["O", "O"]
    result = format_entities(tokens, tags)
    assert result == ""


def test_format_entities_all_entities() -> None:
    tokens = ["Jakarta", "Bandung"]
    tags = ["B-LOC", "B-LOC"]
    result = format_entities(tokens, tags)
    assert "LOC: Jakarta" in result
    assert "LOC: Bandung" in result


def test_parse_generated_entities_basic() -> None:
    text = "PER: Joko, LOC: Jakarta"
    result = parse_generated_entities(text)
    assert result == [("PER", "Joko"), ("LOC", "Jakarta")]


def test_parse_generated_entities_single() -> None:
    text = "PER: Joko"
    result = parse_generated_entities(text)
    assert result == [("PER", "Joko")]


def test_parse_generated_entities_empty() -> None:
    text = ""
    result = parse_generated_entities(text)
    assert result == []


def test_parse_generated_entities_whitespace() -> None:
    text = "  "
    result = parse_generated_entities(text)
    assert result == []


def test_parse_generated_entities_multiword() -> None:
    text = "ORG: PT Mitra Utama, LOC: Surabaya"
    result = parse_generated_entities(text)
    assert result == [("ORG", "PT Mitra Utama"), ("LOC", "Surabaya")]


def test_ner_nergrit_yaml_loads() -> None:
    yaml_path = Path(__file__).parent.parent / "src/nusabench/tasks/configs/ner_nergrit.yaml"
    config = load_task_config(yaml_path)

    assert config.task == "ner_nergrit"
    assert config.dataset_path == "indonlp/indonlu"
    assert config.dataset_name == "nergrit"
    assert config.test_split == "test"
    assert config.validation_split == "validation"
    assert config.output_type == "generate_until"


def test_ner_nergrit_generation_kwargs() -> None:
    yaml_path = Path(__file__).parent.parent / "src/nusabench/tasks/configs/ner_nergrit.yaml"
    config = load_task_config(yaml_path)

    assert config.generation_kwargs["max_tokens"] == 256
    assert "\n\n" in config.generation_kwargs["stop_sequences"]


def test_ner_nergrit_metrics() -> None:
    yaml_path = Path(__file__).parent.parent / "src/nusabench/tasks/configs/ner_nergrit.yaml"
    config = load_task_config(yaml_path)

    assert len(config.metric_list) == 1
    assert config.metric_list[0]["metric"] == "f1"
    assert config.metric_list[0]["aggregation"] == "entity_level"


def test_ner_nergrit_metadata() -> None:
    yaml_path = Path(__file__).parent.parent / "src/nusabench/tasks/configs/ner_nergrit.yaml"
    config = load_task_config(yaml_path)

    assert config.metadata["source"] == "IndoNLU"
    assert config.metadata["language"] == "id"
    assert config.metadata["task_type"] == "sequence_labeling"
    assert "PER" in config.metadata["entity_types"]
    assert "LOC" in config.metadata["entity_types"]
    assert "ORG" in config.metadata["entity_types"]


def test_ner_nergrit_doc_to_text_template() -> None:
    yaml_path = Path(__file__).parent.parent / "src/nusabench/tasks/configs/ner_nergrit.yaml"
    config = load_task_config(yaml_path)
    task = Task(config)

    doc = {"tokens_joined": "Joko tinggal di Jakarta"}
    prompt = task.format_prompt(doc)

    assert "Teks: Joko tinggal di Jakarta" in prompt
    assert "Entitas yang ditemukan:" in prompt


def test_ner_nergrit_doc_to_target_template() -> None:
    yaml_path = Path(__file__).parent.parent / "src/nusabench/tasks/configs/ner_nergrit.yaml"
    config = load_task_config(yaml_path)
    task = Task(config)

    doc = {"entities_formatted": "PER: Joko, LOC: Jakarta"}
    target = task.format_target(doc)

    assert target == "PER: Joko, LOC: Jakarta"


def test_ner_nergrit_task_instantiation() -> None:
    yaml_path = Path(__file__).parent.parent / "src/nusabench/tasks/configs/ner_nergrit.yaml"
    config = load_task_config(yaml_path)
    task = Task(config)

    assert task is not None
    assert isinstance(task, Task)
    assert task.config.task == "ner_nergrit"
    assert task.config.dataset_path == "indonlp/indonlu"
    assert task.config.dataset_name == "nergrit"
    assert task.config.output_type == "generate_until"


def test_ner_nergrit_description() -> None:
    yaml_path = Path(__file__).parent.parent / "src/nusabench/tasks/configs/ner_nergrit.yaml"
    config = load_task_config(yaml_path)

    assert config.description != ""
    assert "Named entity recognition" in config.description
    assert "Indonesian" in config.description


def test_ner_nergrit_task_registry_can_be_registered() -> None:
    yaml_path = Path(__file__).parent.parent / "src/nusabench/tasks/configs/ner_nergrit.yaml"
    config = load_task_config(yaml_path)
    task = Task(config)

    assert task is not None
    assert isinstance(task, Task)
    assert task.config.task == "ner_nergrit"
