# -*- coding: utf-8 -*-
"""
统一的大模型调用封装（OpenAI 兼容接口），供各脚本共用。

切换模型只需修改环境变量，无需改动调用代码。

环境变量：
  LLM_API_KEY          必填，API Key。
  LLM_API_BASE         接口地址，默认 https://dashscope.aliyuncs.com/compatible-mode/v1。
  LLM_MODEL            模型名，默认 qwen-max。
  LLM_ENABLE_THINKING  "true" / "false" / 空字符串；空表示不发送该字段（非 DashScope 模型用空）。

用法：
  from llm_client import call_llm
  text = call_llm([{"role": "user", "content": "..."}])
"""
import os
import time

import requests

API_KEY = os.environ.get("LLM_API_KEY", "")
API_BASE = os.environ.get("LLM_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL = os.environ.get("LLM_MODEL", "qwen-max")
ENABLE_THINKING = os.environ.get("LLM_ENABLE_THINKING", "false")


def call_llm(messages, temperature=0.3, max_tokens=4096, max_retries=3, timeout=120):
    if not API_KEY:
        raise RuntimeError("未设置 LLM_API_KEY。先 export LLM_API_KEY=sk-xxx")
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if ENABLE_THINKING.strip() != "":
        payload["enable_thinking"] = ENABLE_THINKING.strip().lower() == "true"
    url = f"{API_BASE.rstrip('/')}/chat/completions"
    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=timeout)
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            content = (msg.get("content") or "").strip()
            if not content:
                raise RuntimeError("LLM 返回空 content")
            return content
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)  # 1, 2, 4 秒退避
    raise RuntimeError(f"LLM 调用失败（重试 {max_retries} 次）: {last_err}")
