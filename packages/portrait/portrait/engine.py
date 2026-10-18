"""Turn a feature-base job directory into a portrait report."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Optional

import yaml

from .llm import DeepSeek, classify_segments, write_narratives

BIN_MS = 100
DISCLAIMER = (
    "本页指数描述这段录像相对本人基线的行为偏离，仅供访谈复盘参考，"
    "不构成心理诊断、测谎或风险裁决。"
)

UNAVAILABLE = [
    ("perfusion", "面部血流灌注", "v1 未提取 rPPG"),
    ("jitter", "声学抖动", "基座没有抖动专项"),
    ("formant", "共振峰稳定性", "基座没有共振峰专项"),
    ("hr", "静息心率", "v1 未提取心率"),
    ("hrv", "心率变异", "v1 未提取 HRV"),
    ("resp", "呼吸频率", "v1 未提取呼吸"),
    ("rppg_q", "rPPG 信号质量", "v1 未提取 rPPG"),
    ("pupil_d", "瞳孔直径", "只有瞳孔位置，没有直径"),
    ("hr_decel", "心率减速", "v1 未提取心率"),
]

HEATMAP_MARKERS = [
    ("gaze_yaw", "视线水平"),
    ("gaze_pitch", "视线垂直"),
    ("head_yaw", "头部左右"),
    ("head_pitch", "头部俯仰"),
    ("f0_hz", "基频"),
    ("energy_db", "短时能量"),
    ("blink_rate", "眨眼率"),
    ("au_04", "皱眉"),
    ("au_12", "嘴角上扬"),
    ("anger_cluster", "愤怒动作簇"),
    ("polarity", "面部极性代理"),
]


def build_portrait(feature_dir: Path, config_dir: Path, *, use_deepseek: bool = False) -> dict[str, Any]:
    feature_dir = Path(feature_dir)
    config_dir = Path(config_dir)
    rules = _yaml(config_dir / "rules-0.1.0.yaml")
    topics_cfg = _yaml(config_dir / "topics-0.1.0.yaml")
    scales = _yaml(config_dir / "indicators-0.1.0.yaml").get("scales") or {}
    loaded = _load_job(feature_dir)
    duration = loaded["duration_ms"]
    n_bins = max(1, math.ceil(duration / BIN_MS))
    series = _build_series(loaded, n_bins)
    stats = {key: _zstats(vals) for key, vals in series.items()}
    segments = _segments(loaded, topics_cfg)
    deepseek_meta: dict[str, Any] = {
        "requested": bool(use_deepseek),
        "used": False,
        "model": None,
        "reason": None,
    }
    client = DeepSeek()
    if use_deepseek and client.configured:
        deepseek_meta["model"] = client.model
        try:
            segments = classify_segments(client, segments, list(topics_cfg.get("labels") or []))
            deepseek_meta["used"] = True
        except Exception as exc:  # noqa: BLE001 — keep a finished report if the API fails
            deepseek_meta["reason"] = f"topic_failed:{exc}"
            segments = _apply_lexicon(segments, topics_cfg)
            for seg in segments:
                seg["topic_source"] = "lexicon_fallback"
    else:
        if use_deepseek and not client.configured:
            deepseek_meta["reason"] = "no_api_key"
        segments = _apply_lexicon(segments, topics_cfg)

    segments = _merge_topics(segments)
    baseline_limited = duration < int(rules.get("min_baseline_sec") or 30) * 1000
    categories = _categories(series, stats, segments, loaded, scales, baseline_limited)
    anomalies = [] if baseline_limited else _anomalies(series, stats, rules)
    cards = _cards(anomalies, segments, loaded, stats, series, rules)
    if use_deepseek and client.configured and cards:
        try:
            write_narratives(client, cards)
            deepseek_meta["used"] = True
            deepseek_meta["narratives"] = "ok"
        except Exception as exc:  # noqa: BLE001
            deepseek_meta["narratives"] = "failed"
            deepseek_meta["reason"] = f"narrative_failed:{exc}"
            for card in cards:
                card["narrative"] = None
                card["narrative_status"] = "未生成"
    else:
        for card in cards:
            card["narrative"] = None
            card["narrative_status"] = "未生成" if use_deepseek else "未请求"
        if not use_deepseek:
            deepseek_meta["narratives"] = "off"
        elif not client.configured:
            deepseek_meta["narratives"] = "未生成"

    heatmap = _heatmap(segments, stats)
    overview = _overview_series(series)
    report = {
        "disclaimer": DISCLAIMER,
        "versions": {
            "indicators": "indicators-0.1.0",
            "rules": rules.get("version") or "rules-0.1.0",
            "topics": topics_cfg.get("version") or "topics-0.1.0",
            "pipeline": (loaded.get("report") or {}).get("pipeline_version"),
            "grid_ms": loaded["grid_ms"],
            "grid_note": loaded["grid_note"],
        },
        "duration_ms": duration,
        "baseline_limited": baseline_limited,
        "asr_text": loaded.get("asr_text") or "",
        "deepseek": deepseek_meta,
        "categories": categories,
        "heatmap": heatmap,
        "alerts": [_alert_row(c) for c in cards if c["level"] in {"watch", "alert"}],
        "cards": cards,
        "segments": [
            {
                "id": s["id"],
                "topic": s["topic"],
                "polarity": s["polarity"],
                "start_ms": s["start_ms"],
                "end_ms": s["end_ms"],
                "time_label": _clock(s["start_ms"]) + "–" + _clock(s["end_ms"]),
                "text": s["text"],
                "topic_source": s.get("topic_source"),
            }
            for s in segments
        ],
        "metrics": loaded["metrics"],
        "events": loaded["event_summary"],
        "series": overview,
        "coverage": loaded.get("coverage") or {},
    }
    return _clean(report)


def _yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config is not a mapping: {path}")
    return data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _num(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _load_job(feature_dir: Path) -> dict[str, Any]:
    report_path = feature_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    source = report.get("source") or {}
    vision = _read_jsonl(feature_dir / "features" / "vision_frames.jsonl")
    audio = _read_jsonl(feature_dir / "features" / "audio_frames.jsonl")
    grid_ms = BIN_MS
    grid_note = "逐帧重采样到 100 毫秒"
    if not vision:
        vision, audio, grid_ms, grid_note = _from_parquet(feature_dir, audio)
    au = _read_jsonl(feature_dir / "features" / "au_frames.jsonl")
    events = _read_jsonl(feature_dir / "features" / "events.jsonl")
    words = _read_jsonl(feature_dir / "process" / "asr_words.jsonl")
    utterances = _read_jsonl(feature_dir / "process" / "utterances.jsonl")
    vad = _read_jsonl(feature_dir / "process" / "vad_segments.jsonl")
    duration = int(source.get("duration_ms") or 0)
    if duration <= 0:
        stamps = [int(r.get("timestamp_ms") or 0) for r in vision + audio + au]
        duration = max(stamps or [0]) + grid_ms
    metrics = []
    for row in report.get("metrics") or []:
        metrics.append(
            {
                "metric_code": row.get("metric_code"),
                "metric_name": row.get("metric_name"),
                "group": row.get("group"),
                "unit": row.get("unit"),
                "status": row.get("status"),
                "mean": row.get("mean"),
                "std": row.get("std"),
                "min": row.get("min"),
                "max": row.get("max"),
            }
        )
    return {
        "report": report,
        "duration_ms": duration,
        "vision": vision,
        "audio": audio,
        "au": au,
        "events": events,
        "words": words,
        "utterances": utterances,
        "vad": vad,
        "asr_text": report.get("asr_text") or "",
        "metrics": metrics,
        "event_summary": report.get("event_summary") or [],
        "coverage": report.get("coverage") or {},
        "grid_ms": grid_ms,
        "grid_note": grid_note,
    }


def _from_parquet(feature_dir: Path, audio: list[dict[str, Any]]) -> tuple[list, list, int, str]:
    path = feature_dir / "features" / "timeseries.parquet"
    if not path.exists():
        return [], audio, BIN_MS, "没有逐帧文件，按空序列处理"
    import pandas as pd

    frame = pd.read_parquet(path)
    vision_rows = []
    audio_rows = []
    for record in frame.to_dict(orient="records"):
        second = _num(record.get("second")) or 0
        stamp = int(second * 1000)
        vision_rows.append(
            {
                "timestamp_ms": stamp,
                "head_yaw": _num(record.get("head_yaw")),
                "head_pitch": _num(record.get("head_pitch")),
                "head_roll": _num(record.get("head_roll")),
                "gaze_yaw": _num(record.get("gaze_yaw")),
                "gaze_pitch": _num(record.get("gaze_pitch")),
                "ear_left": _num(record.get("ear_left")),
                "ear_right": _num(record.get("ear_right")),
                "pupil_left_x": _num(record.get("pupil_left_x")),
                "pupil_left_y": _num(record.get("pupil_left_y")),
            }
        )
        audio_rows.append(
            {
                "timestamp_ms": stamp,
                "energy_db": _num(record.get("energy_db")),
                "f0_hz": _num(record.get("f0_hz")),
            }
        )
    return vision_rows, audio_rows or audio, 1000, "缺少逐帧文件，退回秒级汇总"


def _bin_mean(rows: list[dict[str, Any]], key: str, n_bins: int) -> list[Optional[float]]:
    buckets: list[list[float]] = [[] for _ in range(n_bins)]
    for row in rows:
        value = _num(row.get(key))
        if value is None:
            continue
        stamp = int(row.get("timestamp_ms") or 0)
        index = min(n_bins - 1, max(0, stamp // BIN_MS))
        buckets[index].append(value)
    return [sum(bucket) / len(bucket) if bucket else None for bucket in buckets]


def _mean_of(rows: list[Optional[float]]) -> Optional[float]:
    vals = [v for v in rows if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _build_series(loaded: dict[str, Any], n_bins: int) -> dict[str, list[Optional[float]]]:
    vision_keys = ["head_yaw", "head_pitch", "head_roll", "gaze_yaw", "gaze_pitch", "pupil_left_x", "pupil_left_y"]
    audio_keys = ["energy_db", "f0_hz"]
    au_keys = ["au_01", "au_04", "au_05", "au_07", "au_09", "au_10", "au_12", "au_15", "au_23", "au_24"]
    series = {key: _bin_mean(loaded["vision"], key, n_bins) for key in vision_keys}
    series.update({key: _bin_mean(loaded["audio"], key, n_bins) for key in audio_keys})
    series.update({key: _bin_mean(loaded["au"], key, n_bins) for key in au_keys})
    series["anger_cluster"] = [
        _mean_of([series["au_04"][i], series["au_05"][i], series["au_07"][i], series["au_23"][i]])
        for i in range(n_bins)
    ]
    polarity: list[Optional[float]] = []
    for i in range(n_bins):
        smile = series["au_12"][i]
        brow = _mean_of([series["au_04"][i], series["au_15"][i]])
        if smile is None and brow is None:
            polarity.append(None)
        else:
            polarity.append((0.0 if smile is None else smile) - (0.0 if brow is None else brow))
    series["polarity"] = polarity
    series["blink_rate"] = _blink_rate(loaded["events"], n_bins)
    series["head_speed"] = _speed(series["head_yaw"], series["head_pitch"])
    series["gaze_speed"] = _speed(series["gaze_yaw"], series["gaze_pitch"])
    series["pitch_speed"] = _axis_speed(series["head_pitch"])
    return series


def _blink_rate(events: list[dict[str, Any]], n_bins: int) -> list[Optional[float]]:
    times = [int(event.get("start_ms") or 0) for event in events if event.get("type") == "blink"]
    out: list[Optional[float]] = []
    for index in range(n_bins):
        center = index * BIN_MS + BIN_MS / 2
        count = sum(1 for stamp in times if center - 2500 <= stamp <= center + 2500)
        out.append(count * 12.0)
    return out


def _speed(a: list[Optional[float]], b: list[Optional[float]]) -> list[Optional[float]]:
    out: list[Optional[float]] = [None]
    for index in range(1, len(a)):
        if a[index] is None or a[index - 1] is None or b[index] is None or b[index - 1] is None:
            out.append(None)
            continue
        delta = math.hypot(a[index] - a[index - 1], b[index] - b[index - 1])
        out.append(delta / (BIN_MS / 1000))
    return out


def _axis_speed(values: list[Optional[float]]) -> list[Optional[float]]:
    out: list[Optional[float]] = [None]
    for index in range(1, len(values)):
        if values[index] is None or values[index - 1] is None:
            out.append(None)
        else:
            out.append(abs(values[index] - values[index - 1]) / (BIN_MS / 1000))
    return out


def _zstats(values: list[Optional[float]]) -> dict[str, Any]:
    valid = [value for value in values if value is not None]
    if len(valid) < 8:
        return {"mean": _mean_of(values), "std": None, "z": [None] * len(values), "ok": False}
    mean = sum(valid) / len(valid)
    var = sum((value - mean) ** 2 for value in valid) / len(valid)
    std = math.sqrt(var)
    if std < 1e-6:
        z = [0.0 if value is not None else None for value in values]
    else:
        z = [(value - mean) / std if value is not None else None for value in values]
    return {"mean": mean, "std": std, "z": z, "ok": True}


def _segments(loaded: dict[str, Any], topics_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    words = loaded["words"]
    chunks = _chunks(words) if words else []
    if not chunks:
        chunks = [
            {
                "start_ms": int(utt.get("start_ms") or 0),
                "end_ms": int(utt.get("end_ms") or 0),
                "text": str(utt.get("text") or "").strip(),
            }
            for utt in loaded["utterances"]
            if str(utt.get("text") or "").strip()
        ]
    if not chunks and loaded.get("asr_text"):
        chunks = [{"start_ms": 0, "end_ms": loaded["duration_ms"], "text": loaded["asr_text"]}]
    if not chunks:
        chunks = [{"start_ms": 0, "end_ms": loaded["duration_ms"], "text": ""}]
    segments = []
    for index, chunk in enumerate(chunks):
        segments.append(
            {
                "id": f"s{index}",
                "start_ms": int(chunk["start_ms"]),
                "end_ms": max(int(chunk["end_ms"]), int(chunk["start_ms"]) + 1),
                "text": chunk["text"],
                "topic": "其他",
                "polarity": "中",
                "topic_source": "pending",
            }
        )
    return segments


def _chunks(words: list[dict[str, Any]], gap_ms: int = 800, max_ms: int = 12000) -> list[dict[str, Any]]:
    usable = [word for word in words if str(word.get("text") or "").strip()]
    if not usable:
        return []
    groups = [[usable[0]]]
    for word in usable[1:]:
        gap = int(word.get("start_ms") or 0) - int(groups[-1][-1].get("end_ms") or 0)
        span = int(word.get("end_ms") or 0) - int(groups[-1][0].get("start_ms") or 0)
        if gap >= gap_ms or span >= max_ms:
            groups.append([word])
        else:
            groups[-1].append(word)
    chunks = []
    for group in groups:
        text = "".join(str(word.get("text") or "").strip() for word in group)
        chunks.append(
            {
                "start_ms": int(group[0].get("start_ms") or 0),
                "end_ms": int(group[-1].get("end_ms") or 0),
                "text": text,
            }
        )
    return chunks


def _apply_lexicon(segments: list[dict[str, Any]], topics_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    lexicon = topics_cfg.get("lexicon") or {}
    polarity_cfg = topics_cfg.get("polarity") or {}
    out = []
    for seg in segments:
        merged = dict(seg)
        if merged.get("topic_source") == "deepseek":
            out.append(merged)
            continue
        merged["topic"] = _topic_label(merged.get("text") or "", lexicon)
        merged["polarity"] = _polarity_label(merged.get("text") or "", polarity_cfg)
        merged["topic_source"] = "lexicon"
        out.append(merged)
    return out


def _topic_label(text: str, lexicon: dict[str, Any]) -> str:
    best, score = "其他", 0
    for label, words in lexicon.items():
        hits = sum(text.count(str(word)) for word in words or [])
        if hits > score:
            best, score = str(label), hits
    return best


def _polarity_label(text: str, polarity_cfg: dict[str, Any]) -> str:
    negative = sum(text.count(str(word)) for word in polarity_cfg.get("negative") or [])
    positive = sum(text.count(str(word)) for word in polarity_cfg.get("positive") or [])
    if any(phrase in text for phrase in ("没事", "还好", "挺好", "没关系")):
        positive += 1
    if negative > positive and negative > 0:
        return "负"
    if positive > negative and positive > 0:
        return "正"
    return "中"


def _merge_topics(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not segments:
        return []
    merged = [dict(segments[0])]
    for seg in segments[1:]:
        prev = merged[-1]
        gap = int(seg["start_ms"]) - int(prev["end_ms"])
        if seg["topic"] not in {"其他", ""} and seg["topic"] == prev["topic"] and gap <= 1000:
            prev["end_ms"] = seg["end_ms"]
            prev["text"] = (prev["text"] + seg["text"]).strip()
            if seg.get("polarity") != "中":
                prev["polarity"] = seg["polarity"]
            continue
        merged.append(dict(seg))
    for index, seg in enumerate(merged):
        seg["id"] = f"s{index}"
    return merged


def _mean(values: list[Optional[float]]) -> Optional[float]:
    nums = [value for value in values if value is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _std(values: list[Optional[float]]) -> Optional[float]:
    nums = [value for value in values if value is not None]
    if len(nums) < 2:
        return None
    avg = sum(nums) / len(nums)
    return math.sqrt(sum((value - avg) ** 2 for value in nums) / len(nums))


def _p90(values: list[Optional[float]]) -> Optional[float]:
    nums = sorted(value for value in values if value is not None)
    if len(nums) < 8:
        return None
    return nums[int(0.9 * (len(nums) - 1))]


def _clamp(value: float) -> float:
    return max(0.0, min(100.0, value))


def _high(value: Optional[float], high: float) -> Optional[float]:
    if value is None or high <= 0:
        return None
    return _clamp(value / high * 100)


def _low(value: Optional[float], low: float, typical: float) -> Optional[float]:
    if value is None or typical <= low:
        return None
    if value >= typical:
        return 0.0
    if value <= low:
        return 100.0
    return _clamp((typical - value) / (typical - low) * 100)


def _spike(stat: dict[str, Any], z_high: float) -> Optional[float]:
    if not stat.get("ok"):
        return None
    peak = _p90(stat["z"])
    if peak is None or z_high <= 0:
        return None
    return _clamp(max(0.0, peak) / z_high * 100)


def _avg(parts: list[Optional[float]]) -> Optional[float]:
    nums = [part for part in parts if part is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _window(series: list[Optional[float]], start_ms: int, end_ms: int) -> list[Optional[float]]:
    i0 = max(0, start_ms // BIN_MS)
    i1 = min(len(series), max(i0 + 1, math.ceil(end_ms / BIN_MS)))
    return series[i0:i1]


def _evidence(code: str, name: str, series: list[Optional[float]], stat: Optional[dict[str, Any]] = None) -> Optional[dict[str, Any]]:
    avg = _mean(series)
    if avg is None:
        return None
    item: dict[str, Any] = {"code": code, "name": name, "mean": round(avg, 3)}
    if stat and stat.get("ok"):
        peak = _p90(stat["z"])
        if peak is not None:
            item["z_p90"] = round(peak, 2)
    return item


def _sub(sub_id: str, name: str, score: Optional[float], *, status: str = "ok", evidence: Optional[list] = None, note: Optional[str] = None) -> dict[str, Any]:
    evidence = [item for item in (evidence or []) if item]
    evidence.sort(key=lambda item: abs(item.get("z_p90") or item.get("mean") or 0), reverse=True)
    if status == "not_available":
        final = None
        state = "not_available"
    elif score is None:
        final = None
        state = "insufficient_data"
    else:
        final = round(float(score), 1)
        state = status
    return {"id": sub_id, "name": name, "score": final, "status": state, "evidence": evidence[:3], "note": note}


def _missing(sub_id: str, name: str, note: str) -> dict[str, Any]:
    return _sub(sub_id, name, None, status="not_available", note=note)


def _chars(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", text or ""))


def _speech_cpm(loaded: dict[str, Any]) -> Optional[float]:
    speech_ms = 0
    for seg in loaded["vad"]:
        if seg.get("label") == "speech":
            speech_ms += int(seg.get("duration_ms") or 0)
    if speech_ms <= 0:
        speech_ms = int((loaded.get("coverage") or {}).get("speech_duration_ms") or 0)
    chars = _chars(loaded.get("asr_text") or "")
    if speech_ms <= 0 or chars <= 0:
        return None
    return chars / (speech_ms / 60000)


def _pause_ratio(loaded: dict[str, Any]) -> Optional[float]:
    if loaded["duration_ms"] <= 0:
        return None
    speech_ms = sum(int(seg.get("duration_ms") or 0) for seg in loaded["vad"] if seg.get("label") == "speech")
    if not loaded["vad"]:
        return None
    return max(0.0, 1 - speech_ms / loaded["duration_ms"])


def _has_any(series: list[Optional[float]]) -> bool:
    return any(value is not None for value in series)


def _micro_per_min(au_frames: list[dict[str, Any]], duration_ms: int) -> tuple[Optional[float], str]:
    if len(au_frames) < 8 or duration_ms <= 0:
        return None, "没有 AU 帧"
    stamps = sorted(int(frame.get("timestamp_ms") or 0) for frame in au_frames)
    gaps = [b - a for a, b in zip(stamps, stamps[1:]) if b > a]
    median_gap = sorted(gaps)[len(gaps) // 2] if gaps else 999
    codes = ["au_01", "au_04", "au_12", "au_15", "au_23"]
    pulses = 0
    for code in codes:
        values = [(int(frame.get("timestamp_ms") or 0), _num(frame.get(code))) for frame in au_frames]
        start = None
        base = None
        for stamp, value in values:
            if value is None:
                continue
            if start is None:
                base = value
                continue
            if base is not None and value - base >= 0.15 and start is None:
                start = stamp
                base = value
            if start is not None and base is not None and value <= base - 0.1:
                if stamp - start <= 200:
                    pulses += 1
                start = None
            base = value
    # The loop above mishandles start. Count short peaks directly.
    pulses = 0
    for code in codes:
        rows = [(int(frame.get("timestamp_ms") or 0), _num(frame.get(code))) for frame in au_frames]
        rows = [(stamp, value) for stamp, value in rows if value is not None]
        for index in range(1, len(rows) - 1):
            prev_v = rows[index - 1][1]
            cur_v = rows[index][1]
            if cur_v - prev_v < 0.12:
                continue
            for later in range(index + 1, len(rows)):
                if rows[later][0] - rows[index - 1][0] > 200:
                    break
                if rows[later][1] <= prev_v + 0.04:
                    pulses += 1
                    break
    per_min = pulses / (duration_ms / 60000)
    note = "AU 帧间隔较大，微表情置信低" if median_gap > 100 else "按短于 200 毫秒的 AU 脉冲计数"
    return per_min, note


def _categories(series, stats, segments, loaded, scales, baseline_limited) -> list[dict[str, Any]]:
    s = scales
    z_high = float(s.get("spike_z_high") or 2.5)
    cpm = _speech_cpm(loaded)
    pause = _pause_ratio(loaded)
    au_ready = _has_any(series["au_04"]) or _has_any(series["au_12"])
    activity = [
        _mean_of([series[code][i] for code in ("au_01", "au_04", "au_06", "au_07", "au_12", "au_15") if code in series])
        for i in range(len(series["head_yaw"]))
    ]
    # au_06 may be absent from series. activity uses existing keys only.
    au_codes = [code for code in ("au_01", "au_04", "au_07", "au_12", "au_15") if code in series]
    activity = [_mean_of([series[code][i] for code in au_codes]) for i in range(len(series["head_yaw"]))]
    micro, micro_note = _micro_per_min(loaded["au"], loaded["duration_ms"])
    note_base = "样本短于 30 秒，突变项未启用" if baseline_limited else None

    mental = [
        _sub(
            "flat_face",
            "表情平坦",
            _avg([_low(_std(activity), 0.02, 0.08), _low(_mean(activity), float(s.get("au_low_flat") or 0.04), float(s.get("au_typical") or 0.16))]) if au_ready else None,
            status="ok" if au_ready else "not_available",
            evidence=[_evidence("au_activity", "AU 活跃度", activity)],
            note=None if au_ready else "没有 AU",
        ),
        _sub(
            "micro",
            "微表情频次",
            _high(micro, float(s.get("micro_per_min_high") or 12)) if micro is not None else None,
            status="ok" if micro is not None else "not_available",
            note=micro_note,
        ),
        _sub(
            "au_low",
            "AU 活跃偏低",
            _low(_mean(activity), float(s.get("au_low_flat") or 0.04), float(s.get("au_typical") or 0.16)) if au_ready else None,
            status="ok" if au_ready else "not_available",
            evidence=[_evidence("au_activity", "AU 活跃度", activity)],
            note=None if au_ready else "没有 AU",
        ),
        _missing("perfusion", "面部血流灌注", "v1 未提取 rPPG"),
        _sub("f0_flat", "语调平坦", _low(_std(series["f0_hz"]), float(s.get("f0_std_flat_hz") or 18), float(s.get("f0_std_typical_hz") or 40)), evidence=[_evidence("f0_hz", "基频", series["f0_hz"], stats["f0_hz"])]),
        _sub("slow_speech", "语速偏低", _low(cpm, float(s.get("speech_cpm_slow") or 90), float(s.get("speech_cpm_typical") or 200)), evidence=[{"code": "speech_rate_cpm", "name": "语速", "mean": None if cpm is None else round(cpm, 1)}] if cpm is not None else []),
        _sub("pause", "停顿比例偏高", _high(pause, float(s.get("pause_ratio_high") or 0.5)), evidence=[{"code": "pause_ratio", "name": "非语音占比", "mean": None if pause is None else round(pause, 3)}] if pause is not None else []),
        _missing("jitter", "声学抖动", "基座没有抖动专项"),
        _missing("formant", "共振峰稳定性", "基座没有共振峰专项"),
        _missing("hr", "静息心率", "v1 未提取心率"),
        _missing("hrv", "心率变异", "v1 未提取 HRV"),
        _missing("resp", "呼吸", "v1 未提取呼吸"),
        _missing("rppg_q", "rPPG 信号质量", "v1 未提取 rPPG"),
    ]
    emotion = [
        _sub(
            "fluctuation",
            "情绪波动",
            _high(_std(series["polarity"]), float(s.get("polarity_std_high") or 0.12)) if au_ready else None,
            status="ok" if au_ready else "not_available",
            evidence=[_evidence("polarity", "面部极性代理", series["polarity"], stats["polarity"])],
            note="极性用 AU12 相对 AU4/AU15，是面部动作代理" if au_ready else "没有 AU",
        ),
        _sub("process", "主题间起伏", _topic_swing(segments, stats), note=note_base),
    ]
    anxiety = [
        _au_level_sub("worry_brow", "担忧眉", [series["au_01"], series["au_04"]], ["au_01", "au_04"], ["内眉", "皱眉"], stats, s, z_high),
        _au_level_sub("alert_eye", "警惕眼", [series["au_05"], _abs(series["gaze_yaw"])], ["au_05", "gaze_yaw"], ["上眼睑", "视线偏离"], stats, s, z_high, highs=[float(s.get("au_high") or 0.35), float(s.get("gaze_focus_deg") or 8)]),
        _au_level_sub("tense_mouth", "紧张口唇", [series["au_23"], series["au_24"]], ["au_23", "au_24"], ["唇收紧", "唇收紧2"], stats, s, z_high),
        _sub(
            "emotion_var",
            "情绪波动",
            _high(_std(series["polarity"]), float(s.get("polarity_std_high") or 0.12)) if au_ready else None,
            status="ok" if au_ready else "not_available",
            note=None if au_ready else "没有 AU",
        ),
        _sub(
            "blink_up",
            "眨眼升高",
            _avg([_high(_mean(series["blink_rate"]), float(s.get("blink_per_min_high") or 28)), _spike(stats["blink_rate"], z_high)]),
            evidence=[_evidence("blink_rate", "眨眼率（次/分）", series["blink_rate"], stats["blink_rate"])],
        ),
        _sub(
            "head_jitter",
            "头部微抖",
            _high(_mean(series["head_speed"]), float(s.get("head_jitter_dps_high") or 12)),
            evidence=[_evidence("head_speed", "头部角速度", series["head_speed"])],
        ),
    ]
    control = [
        _sub("anger_rise", "愤怒飙升", _anger_rise(series["anger_cluster"], float(s.get("anger_rise_high") or 0.25)) if au_ready else None, status="ok" if au_ready else "not_available", note=None if au_ready else "没有 AU"),
        _sub("disgust_co", "厌恶与声学唤醒共现", _coactivation(series, stats) if _has_any(series["au_09"]) or _has_any(series["au_10"]) else None, status="ok" if _has_any(series["au_09"]) or _has_any(series["au_10"]) else "not_available", note="用能量和基频代替心率"),
        _au_level_sub("anger_cluster", "愤怒动作簇", [series["anger_cluster"]], ["anger_cluster"], ["AU4+5+7+23"], stats, s, z_high),
        _sub("face_accel", "动作加速度", _high(_mean(_accel(series["head_yaw"])), 40), evidence=[_evidence("head_yaw", "头部左右", series["head_yaw"])], note="关键点未单独存盘，降级为头部角加速度"),
        _sub("head_thrust", "头部角速度峰值", _high(_p90(series["pitch_speed"]), 25), evidence=[_evidence("pitch_speed", "俯仰角速度", series["pitch_speed"])], note="取俯仰角速度高分位，方向不区分前后"),
        _au_level_sub("lip_tight", "嘴唇紧绷", [series["au_23"], series["au_24"]], ["au_23", "au_24"], ["唇收紧", "唇收紧2"], stats, s, z_high),
    ]
    gaze_abs = [_hypot(series["gaze_yaw"][i], series["gaze_pitch"][i]) for i in range(len(series["gaze_yaw"]))]
    focus_deg = float(s.get("gaze_focus_deg") or 8)
    focused = [value for value in gaze_abs if value is not None]
    focus_ratio = None if not focused else sum(1 for value in focused if value <= focus_deg) / len(focused)
    attention = [
        _sub("gaze_drift", "视线漂移", _high(_mean(series["gaze_speed"]), float(s.get("gaze_speed_dps_high") or 20)), evidence=[_evidence("gaze_speed", "视线速度", series["gaze_speed"])]),
        _sub("focus", "聚焦不足", None if focus_ratio is None else _clamp((1 - focus_ratio) * 100), evidence=[{"code": "focus_ratio", "name": "中心注视占比", "mean": None if focus_ratio is None else round(focus_ratio, 3)}] if focus_ratio is not None else []),
        _missing("hr_decel", "心率减速", "v1 未提取心率"),
    ]
    context = [
        _sub(
            "cognitive",
            "认知负荷",
            _avg([
                _high(_mean(series["blink_rate"]), float(s.get("blink_per_min_high") or 28)),
                _low(cpm, float(s.get("speech_cpm_slow") or 90), float(s.get("speech_cpm_typical") or 200)),
                _high(pause, float(s.get("pause_ratio_high") or 0.5)),
                _high(_mean(gaze_abs), focus_deg),
            ]),
            note="未使用瞳孔直径",
        ),
        _sub("conceal", "掩饰-泄露", _conceal(segments, series, stats), note="语义极性来自词表或 DeepSeek，面部用 AU 代理"),
        _sub("recovery", "应激恢复慢", _recovery(series, stats, float(s.get("recovery_sec_high") or 8), baseline_limited), note=note_base),
        _sub("avoidance", "话题回避", _avoidance(segments, series, loaded, s), note="至少三项同时出现才计分"),
        _sub("micro_density", "微表情密度", _high(micro, float(s.get("micro_per_min_high") or 12)) if micro is not None else None, status="ok" if micro is not None else "not_available", note=micro_note),
        _sub("consistency", "语义-声学唤醒背离", _inconsistency(segments, stats), note="v1 用声学唤醒代替生理唤醒"),
    ]
    named = [
        ("mental", "心理健康信号", mental),
        ("emotion", "情绪波动", emotion),
        ("anxiety", "焦虑信号", anxiety),
        ("control", "反应控制信号", control),
        ("attention", "注意分散", attention),
        ("context", "场景补充信号", context),
    ]
    categories = []
    for cid, name, subs in named:
        scored = [item for item in subs if isinstance(item.get("score"), (int, float))]
        score = round(sum(item["score"] for item in scored) / len(scored), 1) if scored else None
        categories.append({"id": cid, "name": name, "score": score, "coverage": f"{len(scored)}/{len(subs)}", "subs": subs})
    return categories


def _abs(values: list[Optional[float]]) -> list[Optional[float]]:
    return [None if value is None else abs(value) for value in values]


def _hypot(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None and b is None:
        return None
    return math.hypot(0 if a is None else a, 0 if b is None else b)


def _au_level_sub(sub_id, name, rows, codes, names, stats, scales, z_high, highs=None) -> dict[str, Any]:
    present = [row for row in rows if _has_any(row)]
    if not present:
        return _sub(sub_id, name, None, status="not_available", note="没有对应 AU 或视线")
    parts = []
    evidence = []
    for index, row in enumerate(rows):
        if not _has_any(row):
            continue
        high = float(scales.get("au_high") or 0.35)
        if highs and index < len(highs):
            high = highs[index]
        parts.append(_avg([_high(_mean(row), high), _spike(stats.get(codes[index], {"ok": False}), z_high) if codes[index] in stats else None]))
        evidence.append(_evidence(codes[index], names[index], row, stats.get(codes[index])))
    return _sub(sub_id, name, _avg(parts), evidence=evidence)


def _accel(values: list[Optional[float]]) -> list[Optional[float]]:
    speed = _axis_speed(values)
    return _axis_speed(speed)


def _anger_rise(cluster: list[Optional[float]], high: float) -> Optional[float]:
    peak = 0.0
    seen = False
    for index in range(1, len(cluster)):
        if cluster[index] is None or cluster[index - 1] is None:
            continue
        rise = (cluster[index] - cluster[index - 1]) / (BIN_MS / 1000)
        peak = max(peak, rise)
        seen = True
    if not seen:
        return None
    return _high(peak, high)


def _coactivation(series, stats) -> Optional[float]:
    disgust = stats.get("au_09") if stats.get("au_09", {}).get("ok") else stats.get("au_10")
    energy = stats.get("energy_db")
    f0 = stats.get("f0_hz")
    if not disgust or not disgust.get("ok"):
        return None
    hits = 0
    total = 0
    for index, z_dis in enumerate(disgust["z"]):
        if z_dis is None:
            continue
        total += 1
        z_e = energy["z"][index] if energy.get("ok") else None
        z_f = f0["z"][index] if f0.get("ok") else None
        arousal = max([value for value in (z_e, z_f) if value is not None], default=None)
        if z_dis >= 1 and arousal is not None and arousal >= 1:
            hits += 1
    if total == 0:
        return None
    return _clamp(hits / total / 0.08 * 100)


def _topic_swing(segments, stats) -> Optional[float]:
    stat = stats.get("polarity") or {}
    if not stat.get("ok") or len(segments) < 2:
        return None
    means = []
    for seg in segments:
        chunk = _window(stat["z"], seg["start_ms"], seg["end_ms"])
        avg = _mean(chunk)
        if avg is not None:
            means.append(avg)
    if len(means) < 2:
        return None
    turns = sum(1 for a, b in zip(means, means[1:]) if a * b < 0)
    return _clamp(turns / (len(means) - 1) * 100)


def _conceal(segments, series, stats) -> Optional[float]:
    if not segments:
        return None
    conflicts = 0
    considered = 0
    for seg in segments:
        if not seg.get("text"):
            continue
        considered += 1
        pol = seg.get("polarity")
        f0 = _mean(_window(stats["f0_hz"]["z"], seg["start_ms"], seg["end_ms"])) if stats["f0_hz"].get("ok") else None
        au = _mean(_window(stats["polarity"]["z"], seg["start_ms"], seg["end_ms"])) if stats["polarity"].get("ok") else None
        claim_ok = pol == "正" or any(phrase in seg["text"] for phrase in ("没事", "还好", "挺好"))
        aroused = (f0 is not None and f0 >= 0.8) or (au is not None and au <= -0.8)
        smile_over_negative = pol == "负" and au is not None and au >= 0.8
        if (claim_ok and aroused) or smile_over_negative:
            conflicts += 1
    if considered == 0:
        return None
    return _clamp(conflicts / considered * 100)


def _recovery(series, stats, high_sec: float, baseline_limited: bool) -> Optional[float]:
    if baseline_limited:
        return None
    delays = []
    for key in ("f0_hz", "energy_db", "au_04", "anger_cluster", "blink_rate"):
        stat = stats.get(key) or {}
        if not stat.get("ok"):
            continue
        z = stat["z"]
        index = 0
        while index < len(z):
            if z[index] is None or abs(z[index]) < 2:
                index += 1
                continue
            start = index
            while index < len(z) and z[index] is not None and abs(z[index]) >= 2:
                index += 1
            if (index - start) * BIN_MS < 2000:
                continue
            back = index
            while back < len(z) and not (z[back] is not None and abs(z[back]) < 1):
                back += 1
            if back >= len(z):
                delays.append((len(z) - index) * BIN_MS / 1000)
            else:
                delays.append((back - index) * BIN_MS / 1000)
    if not delays:
        return 0.0
    return _high(sum(delays) / len(delays), high_sec)


def _avoidance(segments, series, loaded, scales) -> Optional[float]:
    if not segments:
        return None
    focus_deg = float(scales.get("gaze_focus_deg") or 8)
    slow = float(scales.get("speech_cpm_slow") or 90)
    pause_high = float(scales.get("pause_ratio_high") or 0.5)
    flagged = 0
    for seg in segments:
        signals = 0
        gaze = _mean([_hypot(a, b) for a, b in zip(_window(series["gaze_yaw"], seg["start_ms"], seg["end_ms"]), _window(series["gaze_pitch"], seg["start_ms"], seg["end_ms"]))])
        if gaze is not None and gaze > focus_deg:
            signals += 1
        yaw = _mean(_abs(_window(series["head_yaw"], seg["start_ms"], seg["end_ms"])))
        pitch = _mean(_abs(_window(series["head_pitch"], seg["start_ms"], seg["end_ms"])))
        if (yaw or 0) > 12 or (pitch or 0) > 12:
            signals += 1
        dur = max(1, seg["end_ms"] - seg["start_ms"])
        cpm = _chars(seg["text"]) / (dur / 60000) if seg["text"] else None
        if cpm is not None and cpm < slow:
            signals += 1
        if _segment_pause(loaded["vad"], seg) is not None and _segment_pause(loaded["vad"], seg) > pause_high:
            signals += 1
        if seg["text"] and _chars(seg["text"]) < 12:
            signals += 1
        if signals >= 3:
            flagged += 1
    return _clamp(flagged / len(segments) * 100)


def _segment_pause(vad: list[dict[str, Any]], seg: dict[str, Any]) -> Optional[float]:
    if not vad:
        return None
    span = max(1, seg["end_ms"] - seg["start_ms"])
    silent = 0
    for item in vad:
        if item.get("label") != "non_speech":
            continue
        start = max(seg["start_ms"], int(item.get("start_ms") or 0))
        end = min(seg["end_ms"], int(item.get("end_ms") or 0))
        if end > start:
            silent += end - start
    return silent / span


def _inconsistency(segments, stats) -> Optional[float]:
    if not stats["f0_hz"].get("ok") and not stats["energy_db"].get("ok"):
        return None
    mismatches = 0
    considered = 0
    for seg in segments:
        if seg.get("polarity") not in {"正", "负"}:
            continue
        considered += 1
        parts = []
        for key in ("f0_hz", "energy_db"):
            if stats[key].get("ok"):
                parts.append(_mean(_window(stats[key]["z"], seg["start_ms"], seg["end_ms"])))
        arousal = _mean(parts)
        if arousal is None:
            continue
        sign = 1 if seg["polarity"] == "正" else -1
        if sign * arousal < -0.5:
            mismatches += 1
    if considered == 0:
        return 0.0
    return _clamp(mismatches / considered * 100)


def _covered_ms(values: list[Optional[float]]) -> int:
    indexes = [index for index, value in enumerate(values) if value is not None]
    if not indexes:
        return 0
    if len(indexes) == 1:
        return BIN_MS
    return (indexes[-1] - indexes[0] + 1) * BIN_MS


def _rolling(values: list[Optional[float]], win: int) -> list[Optional[float]]:
    """Centered window: the recording is already complete, so the mark sits on the spike."""
    half = max(0, win // 2)
    out = []
    for index in range(len(values)):
        chunk = [
            values[j]
            for j in range(max(0, index - half), min(len(values), index + half + 1))
            if values[j] is not None
        ]
        out.append(sum(chunk) / len(chunk) if len(chunk) >= max(3, win // 3) else None)
    return out


def _anomalies(series, stats, rules) -> list[dict[str, Any]]:
    names = {code: name for code, name in HEATMAP_MARKERS}
    names.update({"head_roll": "头部倾斜", "energy_db": "短时能量"})
    window = max(1, round(float(rules.get("window_sec") or 5) * 1000 / BIN_MS))
    min_bins = max(1, round(float(rules.get("min_duration_sec") or 2) * 1000 / BIN_MS))
    z_thr = float(rules.get("z_threshold") or 2)
    found = []
    for code in rules.get("markers") or []:
        stat = stats.get(code)
        if not stat or not stat.get("ok") or code not in series:
            continue
        valid_ms = _covered_ms(series[code])
        if valid_ms < int(rules.get("min_baseline_sec") or 30) * 1000:
            continue
        rolled = _rolling(series[code], window)
        mean, std = stat["mean"], stat["std"]
        if std is None or std < 1e-6:
            continue
        mask = []
        z_vals = []
        for value in rolled:
            if value is None:
                mask.append(False)
                z_vals.append(None)
                continue
            z_val = (value - mean) / std
            z_vals.append(z_val)
            mask.append(abs(z_val) >= z_thr)
        index = 0
        while index < len(mask):
            if not mask[index]:
                index += 1
                continue
            end = index
            while end < len(mask) and mask[end]:
                end += 1
            if end - index >= min_bins:
                chunk = [z for z in z_vals[index:end] if z is not None]
                peak = max(chunk, key=abs) if chunk else 0
                found.append(
                    {
                        "code": code,
                        "name": names.get(code, code),
                        "start_ms": index * BIN_MS,
                        "end_ms": end * BIN_MS,
                        "z": peak,
                    }
                )
            index = end
    return found


def _cards(anomalies, segments, loaded, stats, series, rules) -> list[dict[str, Any]]:
    groups = _keyword_groups(loaded["words"], rules)
    used_groups = set()
    drafts = []
    for anomaly in anomalies:
        overlap = [group for group in groups if _overlap(anomaly["start_ms"], anomaly["end_ms"], group["start_ms"], group["end_ms"])]
        for group in overlap:
            used_groups.add(id(group))
        start = anomaly["start_ms"]
        end = anomaly["end_ms"]
        if overlap:
            start = min([start] + [group["start_ms"] for group in overlap])
            end = max([end] + [group["end_ms"] for group in overlap])
        keywords = []
        high_risk = False
        for group in overlap:
            keywords.extend(group["keywords"])
            high_risk = high_risk or group["high_risk"]
        drafts.append(
            {
                "start_ms": start,
                "end_ms": end,
                "keywords": list(dict.fromkeys(keywords)),
                "high_risk": high_risk,
                "marker": anomaly,
                "z": anomaly["z"],
            }
        )
    for group in groups:
        if id(group) in used_groups:
            continue
        drafts.append(
            {
                "start_ms": group["start_ms"],
                "end_ms": group["end_ms"],
                "keywords": group["keywords"],
                "high_risk": group["high_risk"],
                "marker": None,
                "z": 0,
            }
        )
    cards = []
    alert_z = float(rules.get("alert_z") or 2.5)
    for index, draft in enumerate(sorted(drafts, key=lambda item: item["start_ms"])):
        seg = _segment_at(segments, (draft["start_ms"] + draft["end_ms"]) // 2)
        layer1 = _layer1(stats, draft["start_ms"], draft["end_ms"])
        peak = max([abs(item["z"]) for item in layer1], default=abs(draft["z"]))
        repeated = _keyword_repeated(draft["keywords"], seg, loaded["words"])
        has_kw = bool(draft["keywords"])
        has_marker = draft["marker"] is not None
        if has_kw and has_marker and (peak >= alert_z or repeated or draft["high_risk"]):
            level = "alert"
        elif has_marker or (has_kw and peak >= float(rules.get("z_threshold") or 2)):
            level = "watch"
        else:
            level = "info"
        primary = draft["marker"]["code"] if draft["marker"] else (layer1[0]["code"] if layer1 else "f0_hz")
        cards.append(
            {
                "id": f"c{index+1}",
                "level": level,
                "trigger": "both" if has_kw and has_marker else "marker" if has_marker else "keyword",
                "start_ms": draft["start_ms"],
                "end_ms": draft["end_ms"],
                "time_label": _clock(draft["start_ms"]) + "–" + _clock(draft["end_ms"]),
                "keywords": draft["keywords"],
                "topic": None if seg is None else seg["topic"],
                "text": "" if seg is None else seg["text"],
                "layer1": layer1[:5],
                "layer2": _layer2(seg, segments, stats, primary),
                "layer3": _layer3(draft["keywords"], loaded["words"], stats, primary, draft["start_ms"]),
                "curve": _curve(series, primary, draft["start_ms"], draft["end_ms"]),
                "narrative": None,
                "narrative_status": "未生成",
            }
        )
    order = {"alert": 0, "watch": 1, "info": 2}
    cards.sort(key=lambda card: (order.get(card["level"], 9), card["start_ms"]))
    for index, card in enumerate(cards, start=1):
        card["id"] = f"c{index}"
    return cards[:40]


def _keyword_groups(words: list[dict[str, Any]], rules: dict[str, Any]) -> list[dict[str, Any]]:
    catalog = []
    for word in (rules.get("keywords") or {}).get("high_risk") or []:
        catalog.append((str(word), True))
    for word in (rules.get("keywords") or {}).get("concern") or []:
        catalog.append((str(word), False))
    hits = []
    for token in words:
        text = str(token.get("text") or "")
        for keyword, high_risk in catalog:
            if keyword and keyword in text:
                hits.append(
                    {
                        "keyword": keyword,
                        "high_risk": high_risk,
                        "start_ms": int(token.get("start_ms") or 0),
                        "end_ms": int(token.get("end_ms") or 0),
                    }
                )
    if not hits:
        return []
    hits.sort(key=lambda item: item["start_ms"])
    merge_ms = int(float(rules.get("keyword_merge_sec") or 3) * 1000)
    groups = [hits[0:1]]
    for hit in hits[1:]:
        if hit["start_ms"] - groups[-1][-1]["end_ms"] <= merge_ms:
            groups[-1].append(hit)
        else:
            groups.append([hit])
    out = []
    for group in groups:
        out.append(
            {
                "start_ms": group[0]["start_ms"],
                "end_ms": group[-1]["end_ms"],
                "keywords": list(dict.fromkeys(item["keyword"] for item in group)),
                "high_risk": any(item["high_risk"] for item in group),
            }
        )
    return out


def _overlap(a0, a1, b0, b1) -> bool:
    return min(a1, b1) - max(a0, b0) > 0 or abs(((a0 + a1) / 2) - ((b0 + b1) / 2)) < 1500


def _segment_at(segments, stamp: int) -> Optional[dict[str, Any]]:
    for seg in segments:
        if seg["start_ms"] <= stamp <= seg["end_ms"]:
            return seg
    if not segments:
        return None
    return min(segments, key=lambda seg: min(abs(seg["start_ms"] - stamp), abs(seg["end_ms"] - stamp)))


def _layer1(stats, start_ms, end_ms) -> list[dict[str, Any]]:
    rows = []
    for code, name in HEATMAP_MARKERS:
        stat = stats.get(code) or {}
        if not stat.get("ok"):
            continue
        avg = _mean(_window(stat["z"], start_ms, end_ms))
        if avg is None:
            continue
        rows.append({"code": code, "name": name, "z": round(avg, 2), "direction": "偏高" if avg >= 0 else "偏低"})
    rows.sort(key=lambda item: abs(item["z"]), reverse=True)
    return rows


def _layer2(seg, segments, stats, code) -> Optional[dict[str, Any]]:
    stat = stats.get(code) or {}
    if seg is None or not stat.get("ok"):
        return None
    ranked = []
    for item in segments:
        avg = _mean(_window(stat["z"], item["start_ms"], item["end_ms"]))
        if avg is not None:
            ranked.append((item["id"], item["topic"], avg))
    if not ranked:
        return None
    ranked.sort(key=lambda item: abs(item[2]), reverse=True)
    rank = next((index for index, item in enumerate(ranked, start=1) if item[0] == seg["id"]), None)
    current = next(item[2] for item in ranked if item[0] == seg["id"])
    return {
        "topic": seg["topic"],
        "rank": rank,
        "of": len(ranked),
        "z": round(current, 2),
        "marker": code,
    }


def _keyword_repeated(keywords, seg, words) -> bool:
    if not keywords or seg is None:
        return False
    text = "".join(
        str(word.get("text") or "")
        for word in words
        if seg["start_ms"] <= int(word.get("start_ms") or 0) <= seg["end_ms"]
    )
    return any(text.count(keyword) >= 2 for keyword in keywords)


def _layer3(keywords, words, stats, code, this_start) -> list[dict[str, Any]]:
    if not keywords:
        return []
    stat = stats.get(code) or {}
    rows = []
    for token in words:
        text = str(token.get("text") or "")
        if not any(keyword in text for keyword in keywords):
            continue
        start = int(token.get("start_ms") or 0)
        if abs(start - this_start) < 500:
            continue
        z_val = None
        if stat.get("ok"):
            z_val = _mean(_window(stat["z"], start, int(token.get("end_ms") or start)))
        rows.append(
            {
                "time_ms": start,
                "time_label": _clock(start),
                "text": text,
                "z": None if z_val is None else round(z_val, 2),
            }
        )
    return rows[:8]


def _curve(series, code, start_ms, end_ms) -> dict[str, Any]:
    key = code if code in series else "f0_hz"
    pad0 = max(0, start_ms - 3000)
    pad1 = end_ms + 3000
    i0 = pad0 // BIN_MS
    i1 = min(len(series[key]), math.ceil(pad1 / BIN_MS))
    step = 2
    times = []
    values = []
    for index in range(i0, i1, step):
        times.append(index * BIN_MS)
        values.append(series[key][index])
    return {"code": key, "t": times, "values": values}


def _heatmap(segments, stats) -> dict[str, Any]:
    markers = [{"code": code, "name": name} for code, name in HEATMAP_MARKERS]
    z_rows = []
    for seg in segments:
        row = []
        for code, _name in HEATMAP_MARKERS:
            stat = stats.get(code) or {}
            if not stat.get("ok"):
                row.append(None)
                continue
            avg = _mean(_window(stat["z"], seg["start_ms"], seg["end_ms"]))
            row.append(None if avg is None else round(avg, 2))
        z_rows.append(row)
    return {
        "segments": [
            {"id": seg["id"], "label": seg["topic"], "start_ms": seg["start_ms"], "end_ms": seg["end_ms"]}
            for seg in segments
        ],
        "markers": markers,
        "z": z_rows,
    }


def _overview_series(series) -> dict[str, Any]:
    step = 5
    length = len(series["head_yaw"])
    payload: dict[str, Any] = {"t": [index * BIN_MS for index in range(0, length, step)]}
    for code in ("gaze_yaw", "f0_hz", "energy_db", "blink_rate", "au_04", "head_yaw"):
        payload[code] = [series[code][index] if index < len(series[code]) else None for index in range(0, length, step)]
    return payload


def _alert_row(card: dict[str, Any]) -> dict[str, Any]:
    marker = ""
    if card["layer1"]:
        top = card["layer1"][0]
        marker = f"{top['name']}{top['direction']} {top['z']}σ"
    bits = [card["time_label"]]
    if card["keywords"]:
        bits.append("提及「" + "、".join(card["keywords"]) + "」")
    if card["topic"]:
        bits.append(card["topic"])
    if marker:
        bits.append(marker)
    return {"id": card["id"], "level": card["level"], "time_label": card["time_label"], "text": "，".join(bits), "start_ms": card["start_ms"]}


def _clock(ms: int) -> str:
    total = max(0, int(ms // 1000))
    minute, second = divmod(total, 60)
    return f"{minute:02d}:{second:02d}"


def _clean(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 4)
    if isinstance(value, dict):
        return {key: _clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    return value
