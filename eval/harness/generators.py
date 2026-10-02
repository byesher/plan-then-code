"""Pluggable code generators for the eval harness.

Each generator exposes ``generate(prompt: str, problem_id: str) -> str`` and
returns the raw model output (code is extracted later by the harness).

Backends:
- ReferenceGenerator        : returns the reference solution (self-check, no model).
- FileGenerator             : reads a pre-generated solution from disk.
- LocalModelGenerator       : HF transformers, greedy decoding (AutoDL GPU).
- OpenAICompatibleGenerator : any OpenAI-compatible HTTP API (DashScope / vLLM).
"""
from __future__ import annotations

import os
import pathlib


class ReferenceGenerator:
    """Return the reference solution — validates harness + tests without a model."""

    def __init__(self, problems_root: str | pathlib.Path):
        self.problems_root = pathlib.Path(problems_root)

    def generate(self, prompt: str, problem_id: str) -> str:
        ref = self.problems_root / problem_id / "reference.py"
        if not ref.exists():
            raise FileNotFoundError(f"reference.py missing for {problem_id}: {ref}")
        return ref.read_text(encoding="utf-8")


class FileGenerator:
    """Read pre-generated solutions: <solutions_dir>/<problem_id>/solution.py"""

    def __init__(self, solutions_dir: str | pathlib.Path):
        self.solutions_dir = pathlib.Path(solutions_dir)

    def generate(self, prompt: str, problem_id: str) -> str:
        p = self.solutions_dir / problem_id / "solution.py"
        if not p.exists():
            raise FileNotFoundError(f"solution missing: {p}")
        return p.read_text(encoding="utf-8")


class LocalModelGenerator:
    """HF transformers local inference (greedy decoding, temperature ~ 0)."""

    def __init__(self, model_name: str, device: str = "cuda", max_new_tokens: int = 2048,
                 load_in_8bit: bool = False, load_in_4bit: bool = False):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self._model = None
        self._tokenizer = None

    def _load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        kwargs = dict(device_map="auto", trust_remote_code=True)
        if self.load_in_8bit or self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            if self.load_in_4bit:
                import torch
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                )
            else:
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            kwargs["torch_dtype"] = "auto"

        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        self._model.eval()

    def generate(self, prompt: str, problem_id: str) -> str:
        if self._model is None:
            self._load()
        import torch

        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        new = out[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new, skip_special_tokens=True)


class OpenAICompatibleGenerator:
    """OpenAI-compatible chat API (DashScope, vLLM server, DeepSeek, ...)."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 4096, timeout: float = 300.0):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def generate(self, prompt: str, problem_id: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("openai package required: pip install openai") from e
        api_key = self.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
        client = OpenAI(base_url=self.base_url, api_key=api_key or "EMPTY", timeout=self.timeout)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content or ""
