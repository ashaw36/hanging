"""DeepSeek chat client. Sends transcript text and summary numbers only."""

from __future__ import annotations

import json
import os
import re
from typing import Any


class DeepSeek:
    def __init__(self) -> None:
        self.key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        self.base = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
        self.model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip() or "deepseek-chat"

    @property
    def configured(self) -> bool:
        return bool(self.key)

    def chat_json(self, system: str, user: str, timeout: float = 90) -> dict[str, Any]:
        import httpx

        response = httpx.post(
            f"{self.base}/chat/completions",
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=timeout,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return _parse_json(content)


def _parse_json(content: str) -> dict[str, Any]:
    text = content.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("model output is not a JSON object")
    return data


def classify_segments(client: DeepSeek, segments: list[dict[str, Any]], labels: list[str]) -> list[dict[str, Any]]:
    payload = [{"id": s["id"], "text": s["text"][:500]} for s in segments if s.get("text")]
    if not payload:
        return segments
    system = (
        "你是访谈文本标注员。只输出 JSON。"
        "topic 必须是给定标签之一。polarity 只能是 正、中、负。"
        "evidence 必须是原文里的连续短句，不要编造。"
    )
    user = json.dumps({"labels": labels, "segments": payload}, ensure_ascii=False)
    data = client.chat_json(system, user)
    items = data.get("items") or data.get("segments") or []
    by_id = {str(it.get("id")): it for it in items if isinstance(it, dict)}
    allowed = set(labels)
    out = []
    for seg in segments:
        hit = by_id.get(seg["id"])
        topic = str((hit or {}).get("topic") or "其他")
        if topic not in allowed:
            topic = "其他"
        polarity = str((hit or {}).get("polarity") or "中")
        if polarity not in {"正", "中", "负"}:
            polarity = "中"
        merged = dict(seg)
        merged["topic"] = topic
        merged["polarity"] = polarity
        merged["topic_source"] = "deepseek"
        merged["evidence"] = str((hit or {}).get("evidence") or "")[:80]
        out.append(merged)
    return out


def write_narratives(client: DeepSeek, cards: list[dict[str, Any]]) -> None:
    if not cards:
        return
    system = (
        "你根据给定的结构化对比写中文说明，只输出 JSON。"
        "禁止编造输入里没有的心率、次数、秒数和分数。"
        "每张卡片四段：phenomenon、comparison、trend、prediction。"
        "如果输入不足以判断趋势或预测，直接写“现有片段不足以判断”。"
    )
    brief = []
    for card in cards:
        brief.append(
            {
                "id": card["id"],
                "level": card["level"],
                "time": card["time_label"],
                "topic": card.get("topic"),
                "text": (card.get("text") or "")[:180],
                "keywords": card.get("keywords") or [],
                "layer1": card.get("layer1") or [],
                "layer2": card.get("layer2"),
                "layer3": card.get("layer3") or [],
            }
        )
    data = client.chat_json(
        system,
        json.dumps({"cards": brief}, ensure_ascii=False),
        timeout=120,
    )
    items = data.get("cards") or data.get("items") or []
    by_id = {str(it.get("id")): it for it in items if isinstance(it, dict)}
    for card in cards:
        hit = by_id.get(card["id"])
        if not hit:
            card["narrative"] = None
            card["narrative_status"] = "missing"
            continue
        card["narrative"] = {
            "phenomenon": str(hit.get("phenomenon") or "").strip(),
            "comparison": str(hit.get("comparison") or "").strip(),
            "trend": str(hit.get("trend") or "").strip(),
            "prediction": str(hit.get("prediction") or "").strip(),
        }
        card["narrative_status"] = "ok"
