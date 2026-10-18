from __future__ import annotations

import re
from typing import Any, Optional

FILLERS = ["嗯", "啊", "呃", "那个", "就是", "然后", "嗯嗯", "唉"]


def run_asr(wav_path: Optional[str], model_size: str = "tiny") -> dict[str, Any]:
    if not wav_path:
        return {"ok": False, "reason": "no_audio", "words": [], "utterances": [], "text": "", "asr_confidence": None}

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return {"ok": False, "reason": "faster_whisper_missing", "words": [], "utterances": [], "text": "", "asr_confidence": None}

    try:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, info = model.transcribe(wav_path, language="zh", word_timestamps=True, vad_filter=False)
    except Exception as e:
        return {
            "ok": False,
            "reason": f"asr_failed:{e}",
            "words": [],
            "utterances": [],
            "text": "",
            "asr_confidence": None,
        }

    words: list[dict[str, Any]] = []
    utterances: list[dict[str, Any]] = []
    texts: list[str] = []
    confs: list[float] = []

    for seg in segments:
        seg_text = (seg.text or "").strip()
        texts.append(seg_text)
        utterances.append(
            {
                "start_ms": int(round((seg.start or 0) * 1000)),
                "end_ms": int(round((seg.end or 0) * 1000)),
                "text": seg_text,
                "avg_logprob": float(seg.avg_logprob) if seg.avg_logprob is not None else None,
            }
        )
        if seg.words:
            for w in seg.words:
                token = (w.word or "").strip()
                if not token:
                    continue
                # approx confidence from probability if available
                conf = float(getattr(w, "probability", 0.0) or 0.0)
                confs.append(conf)
                words.append(
                    {
                        "text": token,
                        "start_ms": int(round((w.start or 0) * 1000)),
                        "end_ms": int(round((w.end or 0) * 1000)),
                        "confidence": conf,
                    }
                )

    full = "".join(texts)
    # Chinese char count (exclude whitespace/punct lightly)
    chars = len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", full))
    filler_count = 0
    for f in FILLERS:
        filler_count += full.count(f)

    return {
        "ok": True,
        "language": getattr(info, "language", "zh"),
        "words": words,
        "utterances": utterances,
        "text": full,
        "char_count": chars,
        "filler_count": filler_count,
        "asr_confidence": float(sum(confs) / len(confs)) if confs else None,
        "asr_word_count": len(words),
        "model_size": model_size,
    }


def speech_rate_cpm(char_count: int, speech_duration_ms: int) -> Optional[float]:
    if speech_duration_ms <= 0:
        return None
    minutes = speech_duration_ms / 60000.0
    if minutes <= 0:
        return None
    return float(char_count / minutes)
