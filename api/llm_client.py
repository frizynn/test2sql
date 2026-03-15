"""Async LLM client functions for text-to-SQL and NLG model calls."""

from __future__ import annotations

import json
import time
from typing import Optional

import httpx


async def call_text2sql_model(client, prompt: str, grammar: str,
                              url: str, timeout: float) -> dict:
    """Call llama-server chat completions with GBNF grammar constraint.

    Returns dict: {content: str, latency_ms: float, error: str | None}
    """
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 256,
        "grammar": grammar,
    }

    start_time = time.time()
    try:
        resp = await client.post(url, json=payload, timeout=timeout)
        latency_ms = (time.time() - start_time) * 1000
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        return {"content": content, "latency_ms": round(latency_ms, 1), "error": None}
    except httpx.ConnectError:
        latency_ms = (time.time() - start_time) * 1000
        return {"content": "", "latency_ms": round(latency_ms, 1), "error": "model_unavailable"}
    except httpx.TimeoutException:
        latency_ms = (time.time() - start_time) * 1000
        return {"content": "", "latency_ms": round(latency_ms, 1), "error": "model_timeout"}
    except httpx.HTTPError:
        latency_ms = (time.time() - start_time) * 1000
        return {"content": "", "latency_ms": round(latency_ms, 1), "error": "model_unavailable"}
    except (KeyError, IndexError, json.JSONDecodeError, ValueError):
        latency_ms = (time.time() - start_time) * 1000
        return {"content": "", "latency_ms": round(latency_ms, 1), "error": "model_parse_error"}


async def call_nlg_model(client, prompt: str, url: str, timeout: float) -> dict:
    """Call NLG llama-server chat completions (no grammar, temperature 0.1).

    Returns dict: {content: str, latency_ms: float, error: str | None}
    """
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 512,
    }

    start_time = time.time()
    try:
        resp = await client.post(url, json=payload, timeout=timeout)
        latency_ms = (time.time() - start_time) * 1000
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        return {"content": content, "latency_ms": round(latency_ms, 1), "error": None}
    except httpx.ConnectError:
        latency_ms = (time.time() - start_time) * 1000
        return {"content": "", "latency_ms": round(latency_ms, 1), "error": "model_unavailable"}
    except httpx.TimeoutException:
        latency_ms = (time.time() - start_time) * 1000
        return {"content": "", "latency_ms": round(latency_ms, 1), "error": "model_timeout"}
    except httpx.HTTPError:
        latency_ms = (time.time() - start_time) * 1000
        return {"content": "", "latency_ms": round(latency_ms, 1), "error": "model_unavailable"}
    except (KeyError, IndexError, json.JSONDecodeError, ValueError):
        latency_ms = (time.time() - start_time) * 1000
        return {"content": "", "latency_ms": round(latency_ms, 1), "error": "model_parse_error"}


async def call_model_raw(client, messages: list[dict], url: str, timeout: float,
                         temperature: float = 0, max_tokens: int = 256,
                         grammar: Optional[str] = None) -> dict:
    """Flexible LLM call with caller-controlled messages, temperature, and optional grammar.

    Used by voting (temperature=0.8), schema-linking (no grammar, max_tokens=64),
    and decomposition (no grammar) pipelines that need non-standard LLM parameters.

    Returns dict: {content: str, latency_ms: float, error: str | None}
    """
    payload: dict = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if grammar is not None:
        payload["grammar"] = grammar

    start_time = time.time()
    try:
        resp = await client.post(url, json=payload, timeout=timeout)
        latency_ms = (time.time() - start_time) * 1000
        data = resp.json()
        msg = data["choices"][0]["message"]
        content = msg["content"].strip()
        reasoning = msg.get("reasoning_content", "")
        # Fallback: extract <think> tags from content if server didn't parse them
        if not reasoning and "<think>" in content:
            import re as _re
            think_match = _re.search(r"<think>(.*?)</think>", content, _re.DOTALL)
            if think_match:
                reasoning = think_match.group(1).strip()
                content = content[think_match.end():].strip()
        return {
            "content": content,
            "reasoning": reasoning,
            "latency_ms": round(latency_ms, 1),
            "error": None,
        }
    except httpx.ConnectError:
        latency_ms = (time.time() - start_time) * 1000
        return {"content": "", "reasoning": "", "latency_ms": round(latency_ms, 1), "error": "model_unavailable"}
    except httpx.TimeoutException:
        latency_ms = (time.time() - start_time) * 1000
        return {"content": "", "reasoning": "", "latency_ms": round(latency_ms, 1), "error": "model_timeout"}
    except httpx.HTTPError:
        latency_ms = (time.time() - start_time) * 1000
        return {"content": "", "reasoning": "", "latency_ms": round(latency_ms, 1), "error": "model_unavailable"}
    except (KeyError, IndexError, json.JSONDecodeError, ValueError):
        latency_ms = (time.time() - start_time) * 1000
        return {"content": "", "reasoning": "", "latency_ms": round(latency_ms, 1), "error": "model_parse_error"}
