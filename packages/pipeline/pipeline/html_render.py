from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .aggregate import metric_value
from .copy_map import event_name, reason_text
from .utils import ms_to_clock, ms_to_mmss


def _m(metrics: list[dict], code: str, fmt: str = "{:.2f}", default: str = "—") -> str:
    v = metric_value(metrics, code)
    if v is None:
        return default
    try:
        if fmt.endswith("d") or fmt == "{:d}":
            return fmt.format(int(round(v)))
        return fmt.format(v)
    except Exception:
        return str(v)


def _row(m: dict[str, Any], with_missing: bool = True) -> str:
    if m.get("status") == "not_extracted":
        return ""
    miss = f"<td>{(m['missing_rate']*100):.1f}%</td>" if with_missing and m.get("missing_rate") is not None else ("<td>—</td>" if with_missing else "")
    def f(x):
        return "—" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))
    return (
        f"<tr><td class='name'>{m['metric_name']}</td><td class='unit'>{m['unit']}</td>"
        f"<td>{m.get('n_valid') if m.get('n_valid') is not None else '—'}</td>"
        f"{miss}"
        f"<td>{f(m.get('mean'))}</td><td>{f(m.get('std'))}</td>"
        f"<td>{f(m.get('min'))}</td><td>{f(m.get('median'))}</td><td>{f(m.get('max'))}</td></tr>"
    )


def render_report_html(bundle: dict[str, Any], template_path: Optional[Path] = None) -> str:
    """Generate customer-facing HTML from ReportBundle (standalone, not string-replace brittle)."""
    src = bundle["source"]
    cov = bundle["coverage"]
    proc = bundle["process"]
    metrics = bundle["metrics"]
    media = proc["media"]
    qc = proc.get("qc_segments") or []

    face_rows = "".join(_row(m) for m in metrics if m["group"] == "face_head" and m["status"] != "not_extracted")
    gaze_rows = "".join(_row(m) for m in metrics if m["group"] == "gaze" and m["status"] != "not_extracted")
    acoustic_rows = "".join(
        _row(m, with_missing=False)
        for m in metrics
        if m["group"] == "acoustic" and m["status"] != "not_extracted" and not (m["metric_code"].startswith("mfcc_") and m["metric_code"] != "mfcc_1")
    )
    # mfcc note
    acoustic_rows += (
        "<tr><td class='name'>MFCC 第 2–13 维</td><td class='unit'>—</td><td>—</td>"
        "<td colspan='5' style='text-align:left;color:var(--muted)'>完整数据包中提供各维统计</td></tr>"
    )

    au_rows = "".join(_row(m) for m in metrics if m["group"] == "au" and m.get("status") != "not_extracted")
    au_block = (
        "<div class='empty'>本次分析未包含表情动作单元与基础表情概率。"
        "如需该项数据，可在分析配置中开启对应能力后重新处理。</div>"
        if not au_rows
        else (
            "<div class='table-wrap'><table>"
            "<thead><tr><th>指标</th><th>单位</th><th>有效样本</th><th>缺失比例</th>"
            "<th>均值</th><th>标准差</th><th>最小</th><th>中位</th><th>最大</th></tr></thead>"
            f"<tbody>{au_rows}</tbody>"
            "</table></div>"
        )
    )

    callout = ""
    if qc:
        q = qc[0]
        msg = reason_text(q.get("reason_code", ""), q.get("reason"))
        callout = (
            f"<div class='callout'>{ms_to_mmss(q['start_ms'])}–{ms_to_mmss(q['end_ms'])} "
            f"{msg}，相关指标已按有效样本单独统计。</div>"
        )

    ev_sum_rows = ""
    for s in bundle.get("event_summary") or []:
        typ = s["type"]
        extra = []
        if s.get("rate_per_min") is not None:
            extra.append(f"约 {s['rate_per_min']:.1f} 次/分")
        if s.get("mean_pitch_change_deg") is not None:
            extra.append(f"平均俯仰变化 {s['mean_pitch_change_deg']:.1f}°")
        if s.get("total_duration_ms") is not None and typ == "pause_2s":
            extra.append(f"合计 {s['total_duration_ms']/1000:.1f} s")
        if s.get("count", 0) == 0 and typ == "saccade":
            extra.append("本次未检出")
        mean_dur = "—" if s.get("mean_duration_ms") is None else f"{s['mean_duration_ms']:.0f} ms"
        ev_sum_rows += (
            f"<tr><td class='name'>{event_name(typ)}</td><td>{s.get('count',0)}</td>"
            f"<td>{mean_dur}</td><td style='text-align:left;color:var(--muted)'>{'；'.join(extra) or '—'}</td></tr>"
        )

    ev_detail = ""
    for e in (bundle.get("events_preview") or [])[:20]:
        ev_detail += (
            f"<tr><td>{ms_to_clock(e.get('start_ms'))}</td><td>{ms_to_clock(e.get('end_ms'))}</td>"
            f"<td class='name'>{event_name(e['type'])}</td>"
            f"<td>{e.get('duration_ms') if e.get('duration_ms') is not None else '—'}</td>"
            f"<td style='text-align:left;color:var(--muted)'>—</td></tr>"
        )

    non_speech = metric_value(metrics, "non_speech_duration_ms")
    speech_rate = metric_value(metrics, "speech_rate_cpm")
    utt = metric_value(metrics, "utterance_count")
    filler = metric_value(metrics, "filler_count")
    asr_c = metric_value(metrics, "asr_confidence")
    asr_text = bundle.get("asr_text")
    if not asr_text:
        asr_text = "本次未识别到有效语音文本（可能为静音、噪声或非语言音频）。"

    # load CSS from template if exists
    css = ""
    if template_path and Path(template_path).exists():
        text = Path(template_path).read_text(encoding="utf-8")
        if "<style>" in text and "</style>" in text:
            css = text.split("<style>", 1)[1].split("</style>", 1)[0]

    if not css:
        css = "body{font-family:sans-serif;margin:24px} table{border-collapse:collapse;width:100%} td,th{border:1px solid #ddd;padding:6px}"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>多模态行为特征报告 · {src.get('filename')}</title>
<style>{css}</style>
</head>
<body>
<div class="page">
  <div class="topbar">
    <div>
      <div class="brand">多模态行为特征提取</div>
      <h1>行为特征分析报告</h1>
      <p class="subtitle">基于您上传的视频，自动提取面部、姿态、注视与语音等客观行为数据，供研究与后续分析使用。</p>
    </div>
    <div class="actions">
      <a class="btn secondary" href="./metrics_summary.csv">下载汇总表</a>
      <a class="btn" href="./full.zip">下载完整数据包</a>
    </div>
  </div>
  <div class="meta-strip">
    <span>文件 <strong>{src.get('filename')}</strong></span>
    <span>时长 <strong>{ms_to_mmss(src.get('duration_ms'))}</strong></span>
    <span>画面 <strong>{media.get('width')}×{media.get('height')} · {media.get('fps'):.0f} fps</strong></span>
    <span>完成时间 <strong>{bundle.get('generated_at')}</strong></span>
    <span>分析版本 <strong>{bundle.get('pipeline_version')}</strong></span>
  </div>
  <nav class="toc">
    <a href="#process">处理概况</a><a href="#face">面部与头部</a><a href="#gaze">注视与瞳孔</a>
    <a href="#au">表情动作单元</a><a href="#audio">声学特征</a><a href="#speech">言语内容</a><a href="#events">行为事件</a>
  </nav>

  <section class="card" id="process">
    <h2>处理概况</h2>
    <p class="lead">以下记录本次视频的解析方式与检测覆盖情况，便于您核对数据来源。</p>
    <div class="grid-4">
      <div class="stat"><div class="label">视频信息</div><div class="value">{media.get('video_codec') or 'video'}</div>
        <div class="hint">{media.get('width')}×{media.get('height')} · {media.get('frame_count'):,} 帧</div></div>
      <div class="stat"><div class="label">面部检出</div><div class="value">{cov.get('face_detected_frames'):,} / {src.get('frame_count') or media.get('frame_count'):,}</div>
        <div class="hint">约占全片 {cov.get('face_detected_rate',0)*100:.1f}%</div></div>
      <div class="stat"><div class="label">姿态检出</div><div class="value">{cov.get('pose_detected_frames'):,} / {src.get('frame_count') or media.get('frame_count'):,}</div>
        <div class="hint">约占全片 {cov.get('pose_detected_rate',0)*100:.1f}%</div></div>
      <div class="stat"><div class="label">语音活动</div><div class="value">{ms_to_mmss(cov.get('speech_duration_ms'))}</div>
        <div class="hint">约占全片 {cov.get('speech_time_rate',0)*100:.1f}% · 转写 {cov.get('asr_char_count',0)} 字</div></div>
    </div>
    <div class="grid-4" style="margin-top:12px">
      <div class="stat"><div class="label">分析对象</div><div class="value">主画面人物</div>
        <div class="hint">轨迹连续，中途切换 {proc['track'].get('track_switches',0)} 次</div></div>
      <div class="stat"><div class="label">音频处理</div><div class="value">{media.get('audio_out_sr') or 16000} Hz</div>
        <div class="hint">{"已提取音轨" if media.get("audio_present") else "未检测到音轨"}</div></div>
      <div class="stat"><div class="label">时间对齐</div><div class="value">按秒汇总</div>
        <div class="hint">音画偏移 {proc['align'].get('av_offset_ms',0)} ms</div></div>
      <div class="stat"><div class="label">语音分段</div><div class="value">{proc.get('vad_segment_count',0)} 段</div>
        <div class="hint">含词级时间戳 {proc.get('asr_word_count',0)} 条</div></div>
    </div>
    {callout}
  </section>

  <section class="card" id="face">
    <h2>面部与头部</h2>
    <p class="lead">全片有效样本上的描述统计。角度单位为度（°）。</p>
    <div class="table-wrap"><table>
      <thead><tr><th>指标</th><th>单位</th><th>有效样本</th><th>缺失比例</th><th>均值</th><th>标准差</th><th>最小</th><th>中位</th><th>最大</th></tr></thead>
      <tbody>{face_rows}</tbody>
    </table></div>
  </section>

  <section class="card" id="gaze">
    <h2>注视与瞳孔</h2>
    <p class="lead">视线角度相对眼部几何近似；瞳孔位置为归一化坐标。</p>
    <div class="table-wrap"><table>
      <thead><tr><th>指标</th><th>单位</th><th>有效样本</th><th>缺失比例</th><th>均值</th><th>标准差</th><th>最小</th><th>中位</th><th>最大</th></tr></thead>
      <tbody>{gaze_rows}</tbody>
    </table></div>
  </section>

  <section class="card" id="au">
    <h2>表情动作单元</h2>
    <p class="lead">面部肌肉动作强度（FACS AU）与基础表情概率。</p>
    {au_block}
  </section>

  <section class="card" id="audio">
    <h2>声学特征</h2>
    <p class="lead">基于音频轨道提取的声学统计；基频仅在有效发声片段上计算。</p>
    <div class="table-wrap"><table>
      <thead><tr><th>指标</th><th>单位</th><th>有效样本</th><th>均值</th><th>标准差</th><th>最小</th><th>中位</th><th>最大</th></tr></thead>
      <tbody>{acoustic_rows}</tbody>
    </table></div>
  </section>

  <section class="card" id="speech">
    <h2>言语内容</h2>
    <p class="lead">结合语音活动检测与语音识别得到的言语统计。</p>
    <div class="grid-4">
      <div class="stat"><div class="label">语速</div><div class="value">{f"{speech_rate:.0f} 字/分" if speech_rate is not None else "—"}</div><div class="hint">按语音活动时长计算</div></div>
      <div class="stat"><div class="label">非语音时长</div><div class="value">{ms_to_mmss(non_speech)}</div><div class="hint">占全片 {(non_speech or 0)/max(src.get('duration_ms') or 1,1)*100:.1f}%</div></div>
      <div class="stat"><div class="label">语段 / 填充词</div><div class="value">{int(utt or 0)} / {int(filler or 0)}</div><div class="hint">填充词如「嗯」「那个」</div></div>
      <div class="stat"><div class="label">识别平均置信度</div><div class="value">{f"{asr_c:.2f}" if asr_c is not None else "—"}</div><div class="hint">共识别 {cov.get('asr_char_count',0)} 字</div></div>
    </div>
    <div class="transcript"><strong style="display:block;margin-bottom:6px;font-size:13px;color:var(--muted)">转写摘要</strong>{asr_text}</div>
  </section>

  <section class="card" id="events">
    <h2>行为事件</h2>
    <p class="lead">自动识别的离散行为事件汇总与时间明细。</p>
    <div class="table-wrap" style="margin-bottom:16px"><table>
      <thead><tr><th>事件</th><th>次数</th><th>平均时长</th><th>补充说明</th></tr></thead>
      <tbody>{ev_sum_rows}</tbody>
    </table></div>
    <div class="table-wrap"><table>
      <thead><tr><th>开始时间</th><th>结束时间</th><th>事件</th><th>时长</th><th>说明</th></tr></thead>
      <tbody>{ev_detail}</tbody>
    </table></div>
  </section>

  <footer class="note">{bundle.get('disclaimer','')}</footer>
</div>
</body>
</html>
"""
    return html
