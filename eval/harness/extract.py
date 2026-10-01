"""Extract a Python module source from a model's raw answer."""
from __future__ import annotations

import re

_FENCED = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Return the Python source contained in ``text``.

    Handles both fenced (```python ... ```) and unfenced output.
    """
    if not text:
        return ""
    t = text.strip()
    m = _FENCED.search(t)
    if m:
        return m.group(1).strip()
    # Unclosed fence: drop a leading ```lang line and a trailing ``` if present.
    if t.startswith("```"):
        t = re.sub(r"^```[^\n]*\n?", "", t, count=1)
        t = re.sub(r"\n?```$", "", t)
    return t.strip()
