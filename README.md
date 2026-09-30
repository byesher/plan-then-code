# plan-then-code

Teach a small LLM (Qwen2.5-7B) to **plan before coding** — first emit a structured outline (function signatures + contracts), then implement each function — to improve long / multi-function code generation.

## Idea
Long, multi-function code is where small models break down (interface inconsistency, forgetting constraints). We fine-tune the model on structured `(requirement → outline → implementations)` data via SFT, so it internalizes "plan-then-implement".

## Pipeline
1. **Data** — build `(requirement, outline, function implementations)` triples (synthesis + reverse-engineering from real code)
2. **SFT** — LoRA fine-tune Qwen2.5-7B
3. **Eval** — multi-function module generation → compile rate + test pass rate, before vs after

## Structure
