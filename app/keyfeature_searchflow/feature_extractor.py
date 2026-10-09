"""
Key Feature Extractor: Calls LLM with 3 inputs and returns 5-15 key technical features.
"""

import json
import logging
from typing import Any, cast

from openai import OpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam

from app.config import (
    QUERY_LLM_MAX_TOKENS,
    QUERY_LLM_REMOTE_API_KEY,
    QUERY_LLM_REMOTE_BASE_URL,
    QUERY_LLM_REMOTE_MODEL,
    QUERY_LLM_REQUEST_TIMEOUT,
    QUERY_LLM_TEMPERATURE,
)
from app.keyfeature_searchflow.models import KeyFeatureItem
from app.keyfeature_searchflow.prompts import build_feature_extraction_prompt
from app.query_understanding.engine import repair_truncated_json

logger = logging.getLogger(__name__)


class KeyFeatureExtractor:
    def __init__(self):
        self.base_url = QUERY_LLM_REMOTE_BASE_URL.rstrip("/")
        self.model = QUERY_LLM_REMOTE_MODEL
        self.api_key = QUERY_LLM_REMOTE_API_KEY
        self.timeout = QUERY_LLM_REQUEST_TIMEOUT
        self.temperature = QUERY_LLM_TEMPERATURE
        self.sync_client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=1,
        )

    def _prepare_messages(self, prompt: str) -> list[ChatCompletionMessageParam]:
        return [
            {"role": "user", "content": prompt},
        ]

    def _clean_json_text(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

    def extract_features(
        self,
        problem: str,
        invention_title: str,
        invention_details: str,
    ) -> list[KeyFeatureItem]:
        
        prompt = build_feature_extraction_prompt(
            problem=problem,
            invention_title=invention_title,
            invention_details=invention_details,
        )
        messages = self._prepare_messages(prompt)

        for attempt in range(2):
            print(f"[FeatureLLM] Request -> {self.base_url} (model={self.model})")

            try:
                completion = self.sync_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=self.temperature,
                    timeout=self.timeout,
                    max_tokens=QUERY_LLM_MAX_TOKENS,
                )
                print(
                    f"[FeatureLLM] Response <- {self.base_url} (model={self.model}) | "
                )
                raw_text = completion.choices[0].message.content or "{}"
            except OpenAIError as exc:
                logger.error("LLM communication failed for feature extraction: %s", exc)
                raise RuntimeError(f"Feature extraction LLM request failed: {exc}") from exc

            try:
                return self.parse_features_from_json(raw_text)
            except ValueError:
                if attempt == 0:
                    logger.warning(
                        "LLM returned wrong JSON shape for feature extraction, retrying once"
                    )
                    messages = messages + [
                        {"role": "assistant", "content": raw_text},
                        {
                            "role": "user",
                            "content": (
                                "That response was invalid. Respond again with ONLY a JSON "
                                'object of the exact shape {"key_features": [{"feature_id": 1, '
                                '"feature": "..."}, ...]}. Do not echo the input fields.'
                            ),
                        },
                    ]
                    continue
                raise

        raise RuntimeError("Feature extraction failed after retry")  # unreachable

    def parse_features_from_json(self, raw_text: str) -> list[KeyFeatureItem]:
        cleaned = self._clean_json_text(raw_text)
        try:
            data: dict[str, Any] = json.loads(cleaned)
        except json.JSONDecodeError:
            repaired = repair_truncated_json(cleaned)
            if repaired is None:
                raise ValueError(f"Failed to parse LLM output as JSON: {raw_text[:200]}")
            data = repaired

        raw_features = data.get("key_features", [])
        if not isinstance(raw_features, list) or not raw_features:
            raise ValueError(f"LLM output missing valid 'key_features' array: {data}")
        raw_features = cast(list[Any], raw_features)

        features: list[KeyFeatureItem] = []
        for idx, item in enumerate(raw_features, start=1):
            if isinstance(item, dict):
                item = cast(dict[str, Any], item)
                fid = item.get("feature_id", idx)
                try:
                    fid = int(fid)
                except (ValueError, TypeError):
                    fid = idx
                text = str(item.get("feature", "")).strip()
                if text:
                    features.append(KeyFeatureItem(feature_id=fid, feature=text))
            elif isinstance(item, str) and item.strip():
                features.append(KeyFeatureItem(feature_id=idx, feature=item.strip()))

        if len(features) < 5 or len(features) > 15:
            raise ValueError(
                f"Feature count must be between 5 and 15, got {len(features)}"
            )

        return features

