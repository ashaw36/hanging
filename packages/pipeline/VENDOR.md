# 特征底座

这里的 `pipeline` 包和仓库根目录的 `models/` 来自「多模态行为特征提取基座」0.1.0。

鸿心影鉴在 `report_builder.py` 里多写了两份逐帧文件，供画像层做 100 毫秒对齐：

- `features/vision_frames.jsonl`
- `features/audio_frames.jsonl`

秒级 `timeseries.parquet` 和客户向指标表仍按底座原来的方式产出。
