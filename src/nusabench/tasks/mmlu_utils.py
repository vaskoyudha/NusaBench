from __future__ import annotations


def _index_to_letter(idx: int) -> str:
    letters = ["A", "B", "C", "D"]
    if 0 <= idx < len(letters):
        return letters[idx]
    return ""


def _normalize_answer(ans: object) -> str:
    """Normalize answer to letter A/B/C/D.

    Supports int indices (0-based), numeric strings, or already letter strings.
    """
    if isinstance(ans, int):
        return _index_to_letter(ans)
    try:
        s = str(ans).strip()
    except Exception:
        return ""
    if s.isdigit():
        try:
            return _index_to_letter(int(s))
        except Exception:
            return ""
    # If single letter like 'a' or 'A'
    if len(s) == 1 and s.upper() in "ABCD":
        return s.upper()
    # If it's like 'choice_a' or 'A.' or 'A)'
    if len(s) >= 1:
        up = s.upper()
        for ch in up:
            if ch in "ABCD":
                return ch
    return ""


def expand_choices(doc: dict) -> dict:
    """Expand 'choices' list into choice_a..choice_d and normalize 'answer'.

    If the document already has choice_a keys, leave them intact.
    """
    result = dict(doc)
    choices = doc.get("choices") or []
    # fill choice_a .. choice_d
    for i, letter in enumerate("abcd"):
        key = f"choice_{letter}"
        if key in result:
            # respect existing
            continue
        result[key] = choices[i] if i < len(choices) else ""

    # normalize answer into letter A/B/C/D
    if "answer" in result:
        result["answer"] = _normalize_answer(result["answer"]) or result.get("answer", "")

    return result
