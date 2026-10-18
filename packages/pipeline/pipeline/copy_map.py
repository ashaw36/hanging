"""Customer-facing copy maps. Internal codes must not appear in report.html."""

from __future__ import annotations

REASON_COPY = {
    "QC_PROFILE": "检测到侧脸，注视相关指标可靠度较低",
    "QC_NO_FACE": "未检出面部",
    "QC_AUDIO_NOISE": "音频噪声较大，语音相关指标可能不稳定",
    "QC_OCCLUSION": "面部遮挡，部分面部指标不可靠",
}

EVENT_COPY = {
    "blink": "眨眼",
    "saccade": "眼跳",
    "pause_2s": "停顿 ≥ 2 秒",
    "speech_onset": "开始说话",
    "head_nod": "点头",
    "head_shake": "摇头",
    "smile_onset": "微笑起始",
}


def reason_text(reason_code: str, fallback: str | None = None) -> str:
    if reason_code in REASON_COPY:
        return REASON_COPY[reason_code]
    if fallback:
        return fallback
    return "该时段部分指标可靠度较低"


def event_name(event_type: str) -> str:
    return EVENT_COPY.get(event_type, event_type)
