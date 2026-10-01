"""Build the prompt for a problem from its problem.json."""
from __future__ import annotations

from typing import Any


def _render_interface(interface: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for unit in interface:
        if unit.get("type") == "class":
            sig = f"({unit['signature']})" if unit.get("signature") else ""
            lines.append(f"### Class `{unit['name']}`")
            lines.append(f"Signature: `class {unit['name']}{sig}`")
            lines.append(f"Contract: {unit['contract']}")
            lines.append("Methods:")
            for m in unit.get("methods", []):
                lines.append(f"  - `{m['name']}({m['signature']})` — {m['contract']}")
        else:
            lines.append(f"### Function `{unit['name']}`")
            lines.append(f"Signature: `def {unit['name']}({unit['signature']})`")
            lines.append(f"Contract: {unit['contract']}")
    return "\n".join(lines)


def render_prompt(problem: dict[str, Any]) -> str:
    """Render a 'direct implement' prompt: requirement + fixed interface -> code."""
    interface = _render_interface(problem["interface"])
    return (
        "You are an expert Python programmer. Implement the module described below.\n\n"
        f"# Requirement\n{problem['requirement']}\n\n"
        f"# Required interface (implement exactly; keep names and signatures unchanged)\n{interface}\n\n"
        "# Rules\n"
        "- Output ONLY the Python code, with no explanation and no markdown fences.\n"
        "- You may add private helper functions/classes, but must expose the interface above exactly.\n"
        "- Use only the Python standard library.\n"
        "- The code must be importable as a module named `solution`.\n"
    )
