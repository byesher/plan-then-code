# plan-then-code

Fine-tune a small LLM (Qwen2.5-7B) to **plan before coding** — first generate a structured outline (function signatures + contracts) internally, then implement each function — to improve long / multi-function code generation.

## Why

Small models are strong at short, single-function tasks but break down on long, multi-function programs: they lose track of interfaces, forget constraints, and produce inconsistent implementations.

The hypothesis: a **plan-then-implement** step — writing an outline of function signatures and contracts *before* the implementation — anchors the structure and keeps long code consistent.

## How

**SFT with structured data.** The model is fine-tuned on `(requirement → outline → implementation)` triples: the outline lists each function's signature and contract, the implementation is the actual code. Through SFT the model internalizes "plan first, then code".

**End-to-end evaluation.** The model receives only a natural-language requirement and must write complete, runnable code; we measure whether it *runs and passes the test cases*. The outline is a training-time scaffold only — it never appears in the evaluation input or output, so the score reflects real code quality, not the ability to emit an outline.

**I/O problems as the primary benchmark.** "Write a complete program reading stdin / writing stdout" is the purest end-to-end coding task; interface-mode problems (where signatures are given up front) are a secondary reference.

## Repository

```
eval/    # benchmark & evaluation harness (harness/run_eval.py, benchmark/problems/)
data/    # training-data construction (requirement → outline → implementation)
train/   # SFT (LoRA) fine-tuning
```

See [`eval/README.md`](eval/README.md) for the harness and metrics, and [`eval/analysis.md`](eval/analysis.md) for the interface-vs-I/O evaluation rationale.
