"""LMStudioBackend — calls LM Studio's OpenAI-compatible API via curl."""

from __future__ import annotations

import json
import subprocess

from project_ender.oracle.prompts import AGGRESSIVE_STYLE_HINT

from .base import Backend


class LMStudioBackend(Backend):
    """
    Queries an LM Studio instance via its OpenAI-compatible chat endpoint.

    LM Studio serves on port 1234 by default. For LAN access over Tailscale,
    pass the Tailscale IP as host (e.g. "http://100.x.x.x:1234").

    The model parameter is sent in the request but LM Studio typically
    ignores it and uses whatever model is currently loaded.
    """

    def __init__(
        self,
        model: str = "local-model",
        host: str = "http://localhost:1234",
    ) -> None:
        super().__init__()
        self._model = model
        self._host = host
        self._teacher = f"lmstudio_{model}".replace("/", "_").replace(":", "_")

        # Apply aggressive style hint to models prone to mode collapse
        if model in ("gpt-oss-20b", "deepseek-r1-distill-qwen-14b"):
            self.style_hint = AGGRESSIVE_STYLE_HINT

    @property
    def teacher_id(self) -> str:
        return self._teacher

    def call(self, prompt: str, valid_actions: list[int] | None = None) -> str:
        body: dict = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 512,
        }

        # Constrained decoding: restrict action_id to only valid values
        # and cap reasoning length via the schema.
        if valid_actions:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "oracle_label",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "action_id": {"type": "integer", "enum": valid_actions},
                            "confidence": {"type": "number"},
                            "reasoning": {"type": "string"},
                        },
                        "required": ["action_id", "confidence", "reasoning"],
                        "additionalProperties": False,
                    },
                },
            }

        payload = json.dumps(body)
        result = subprocess.run(
            [
                "curl",
                "-s",
                "-X",
                "POST",
                f"{self._host}/v1/chat/completions",
                "-H",
                "Content-Type: application/json",
                "-d",
                payload,
            ],
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
        )
        data = json.loads(result.stdout)
        message = data["choices"][0]["message"]
        content = message.get("content", "")
        # Reasoning models (e.g. GPT-OSS) put chain-of-thought in a
        # "reasoning" field and may leave "content" empty. Fall back if so.
        if not content.strip() and message.get("reasoning"):
            content = message["reasoning"]
        return content
