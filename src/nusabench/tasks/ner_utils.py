from __future__ import annotations


def join_tokens(tokens: list[str]) -> str:
    """Join BIO-tagged token list into a sentence string."""
    return " ".join(tokens)


def format_entities(tokens: list[str], tags: list[str]) -> str:
    """Extract entities from BIO tags and format as 'TYPE: text' pairs.

    Args:
        tokens: List of token strings
        tags: List of BIO tag strings (e.g. B-PER, I-PER, O)

    Returns:
        Formatted string like "PER: Joko, LOC: Jakarta"
    """
    entities: list[str] = []
    current_entity: list[str] = []
    current_type: str | None = None

    for token, tag in zip(tokens, tags, strict=False):
        if tag.startswith("B-"):
            if current_entity and current_type:
                entities.append(f"{current_type}: {' '.join(current_entity)}")
            current_entity = [token]
            current_type = tag[2:]
        elif tag.startswith("I-") and current_type == tag[2:]:
            current_entity.append(token)
        else:
            if current_entity and current_type:
                entities.append(f"{current_type}: {' '.join(current_entity)}")
            current_entity = []
            current_type = None

    if current_entity and current_type:
        entities.append(f"{current_type}: {' '.join(current_entity)}")

    return ", ".join(entities)


def parse_generated_entities(text: str) -> list[tuple[str, str]]:
    """Parse model output back to entity list.

    Parses strings like "PER: Joko, LOC: Jakarta" back to [("PER", "Joko"), ("LOC", "Jakarta")].
    """
    result: list[tuple[str, str]] = []
    if not text.strip():
        return result
    for part in text.split(","):
        part = part.strip()
        if ":" in part:
            entity_type, entity_text = part.split(":", 1)
            result.append((entity_type.strip(), entity_text.strip()))
    return result
