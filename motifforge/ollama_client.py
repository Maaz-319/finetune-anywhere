"""Tiny Ollama chat client used by the demo and local evaluation."""

from __future__ import annotations

import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from svg_utils import SYSTEM_PROMPT


def chat(base_url, model, prompt, system=SYSTEM_PROMPT, max_new_tokens=2048, temperature=0.0, timeout=900):
    """Send one chat turn to a local Ollama server and return text, seconds, payload."""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": temperature,
            "num_ctx": 4096,
            "num_predict": max_new_tokens,
        },
    }
    request = Request(
        base_url.rstrip("/") + "/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Cannot reach Ollama at {base_url}. Start it with 'ollama serve'.") from exc
    text = (payload.get("message") or {}).get("content", "")
    return text, time.monotonic() - started, payload
