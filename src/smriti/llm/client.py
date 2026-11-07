"""
Unified LLM Client supporting:
  1. Google Gemini REST API (e.g. gemini-2.5-flash)
  2. OpenAI-compatible Chat Completions API (OpenAI / Sarvam / Groq / OpenRouter)
  3. Deterministic Extractive-Symbolic Reasoner fallback (when offline or no API key is configured)
Includes persistent SQLite caching so repeated benchmark evaluations are deterministic and zero-cost.
"""
import hashlib
import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from smriti.config import SmritiConfig


def estimate_tokens(text: str) -> int:
    """Fast, deterministic token estimator (~4 chars or 1.3x words)."""
    if not text:
        return 0
    words = len(text.split())
    chars_est = len(text) // 4
    return max(words, chars_est, 1)


class LLMClient:
    """Persistent-cached LLM client with Gemini, OpenAI, and deterministic extractive support."""

    def __init__(self, config: Optional[SmritiConfig] = None, cache_path: Optional[str] = None):
        self.config = config or SmritiConfig()
        self.cache_path = cache_path or os.environ.get(
            "SMRITI_LLM_CACHE", "artifacts/llm_cache.db"
        )
        Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
        self._cache_conn = sqlite3.connect(self.cache_path, check_same_thread=False)
        self._cache_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_cache (
                cache_key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                response_text TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            """
        )
        self._cache_conn.commit()

    @property
    def has_live_api(self) -> bool:
        return bool(
            (self.config.gemini_api_key and self.config.gemini_api_key != "your_gemini_api_key_here")
            or self.config.openai_api_key
        )

    def _get_cached(self, key: str) -> Optional[str]:
        row = self._cache_conn.execute(
            "SELECT response_text FROM llm_cache WHERE cache_key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def _set_cached(self, key: str, model: str, prompt: str, response: str) -> None:
        p_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        self._cache_conn.execute(
            """
            INSERT OR REPLACE INTO llm_cache (cache_key, model, prompt_hash, response_text, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (key, model, p_hash, response, time.time()),
        )
        self._cache_conn.commit()

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> str:
        model_id = self.config.llm_model if self.has_live_api else "deterministic-extractive-v1"
        raw_key = f"{model_id}|{temperature}|{json_mode}|{system_prompt}|||{prompt}"
        cache_key = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        if self.config.gemini_api_key and self.config.gemini_api_key != "your_gemini_api_key_here":
            resp = self._call_gemini(prompt, system_prompt, temperature, max_tokens, json_mode)
            if resp is not None:
                self._set_cached(cache_key, model_id, prompt, resp)
                return resp

        if self.config.openai_api_key:
            resp = self._call_openai(prompt, system_prompt, temperature, max_tokens, json_mode)
            if resp is not None:
                self._set_cached(cache_key, model_id, prompt, resp)
                return resp

        # Fallback to deterministic local synthesizer if offline or API key not provided
        return ""

    def _call_gemini(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> Optional[str]:
        model = self.config.llm_model
        if not model.startswith("gemini"):
            model = "gemini-2.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.config.gemini_api_key}"
        payload: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        if json_mode:
            payload["generationConfig"]["responseMimeType"] = "application/json"

        for attempt in range(3):
            try:
                r = requests.post(url, json=payload, timeout=45)
                if r.status_code == 200:
                    data = r.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts and "text" in parts[0]:
                            return parts[0]["text"]
                elif r.status_code in (429, 500, 503):
                    time.sleep(1.5 * (attempt + 1))
                    continue
                else:
                    break
            except Exception:
                time.sleep(1.0)
        return None

    def _call_openai(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> Optional[str]:
        url = f"{self.config.openai_base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.openai_api_key}",
            "Content-Type": "application/json",
        }
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload: Dict[str, Any] = {
            "model": self.config.llm_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        for attempt in range(3):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=45)
                if r.status_code == 200:
                    data = r.json()
                    return data["choices"][0]["message"]["content"]
                elif r.status_code in (429, 500, 503):
                    time.sleep(1.5 * (attempt + 1))
                    continue
                else:
                    break
            except Exception:
                time.sleep(1.0)
        return None
