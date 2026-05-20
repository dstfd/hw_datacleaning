"""Thin Gemini wrapper using Vertex AI ADC + the google-genai SDK.

Reads GOOGLE_CLOUD_PROJECT and VERTEX_LOCATION from .env.local (or env).
Provides .json_call() — sends a prompt, returns parsed JSON.
"""

from __future__ import annotations
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parent.parent.parent
ENV_FILE = ROOT / ".env.local"

# Manually parse .env.local (we don't want to pull python-dotenv just for this).
if ENV_FILE.exists():
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("VERTEX_LOCATION", "us-central1")
DEFAULT_MODEL = os.environ.get("LIVE_MODEL", "gemini-2.5-flash")

if not PROJECT:
    raise RuntimeError("GOOGLE_CLOUD_PROJECT not set in env or .env.local")

_client = genai.Client(vertexai=True, project=PROJECT, location=LOCATION)


def _strip_code_fence(text: str) -> str:
    """LLMs sometimes wrap JSON in ```json ... ``` — strip it."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def json_call(
    prompt: str,
    *,
    system: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 8192,
    temperature: float = 0.0,
    retries: int = 3,
) -> Any:
    """Send a prompt, expect JSON back, return parsed JSON.

    Retries on transient errors and on JSON parse failures (re-asks model).
    """
    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            cfg = types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                response_mime_type="application/json",
                system_instruction=system if system else None,
            )
            resp = _client.models.generate_content(
                model=model, contents=prompt, config=cfg
            )
            text = resp.text or ""
            cleaned = _strip_code_fence(text)
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            last_err = f"JSON parse failed: {e}; got: {text[:200] if 'text' in dir() else '<no text>'}"
        except Exception as e:  # noqa: BLE001 — log and retry transient API errors
            last_err = f"{type(e).__name__}: {e}"

        if attempt < retries:
            time.sleep(2 * attempt)

    raise RuntimeError(f"json_call failed after {retries} attempts: {last_err}")


def batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]
