# -*- coding: utf-8 -*-
"""Build 18 self-contained HTML pages from shared CSS/JS sources."""
from pathlib import Path
import json

root = Path(__file__).parent
pages = [
    ("index", "首页工作台"),
    ("analysis", "新建分析任务"),
    ("batch", "历史视频批量导入"),
    ("capture", "现场采集"),
    ("tasks", "解析任务中心"),
    ("data", "数据管理"),
    ("semantic", "语义分析"),
    ("markers", "数字标志物分析"),
    ("alignment", "语义—标志物对齐分析"),
    ("indicators", "预测性指标分析"),
    ("special", "特殊内容识别与对比"),
    ("alerts", "异常突变与预警中心"),
    ("reports", "报告中心"),
    ("logs", "原始数据与可复现日志"),
    ("models", "模型管理与运行状态"),
    ("settings", "系统设置与本地安全"),
    ("profile", "心理画像总览"),
    ("report", "视频分析报告"),
]
css = (root / "base.css").read_text(encoding="utf-8") + "\n" + (root / "system.css").read_text(encoding="utf-8")
js = (
    (root / "api.js").read_text(encoding="utf-8")
    + "\n"
    + (root / "system.js").read_text(encoding="utf-8")
    + "\n"
    + (root / "core.js").read_text(encoding="utf-8")
)
for key, title in pages:
    html = (
        f"<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">"
        f"<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<meta name=\"theme-color\" content=\"#07111e\">"
        f"<meta name=\"description\" content=\"鸿心影鉴 PsyTrace 多模态视频分析系统\">"
        f"<title>{title} · 鸿心影鉴 PsyTrace</title><style>{css}</style></head>"
        f"<body data-page=\"{key}\"><noscript>请启用 JavaScript 查看页面。</noscript>"
        f"<script>{js}</script></body></html>"
    )
    (root / f"{key}.html").write_text(html, encoding="utf-8")
(root / "页面清单.json").write_text(
    json.dumps([{"file": f"{k}.html", "title": t} for k, t in pages], ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(f"Built {len(pages)} HTML pages with api.js + system.js + core.js.")
