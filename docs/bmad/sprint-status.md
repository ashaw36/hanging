# Sprint Status · 原型对接整合

> 更新日期：2026-10-21（报告页三项核心分析指标）  
> 依据：`docs/原型对接-整合方案.md` 第四节 · `docs/功能校验-问题清单.md`

## 已完成

0. **报告页三项核心分析指标**（本轮）  
   - `core.js` `reportPage()`：在 `reportmeta` 与 `chapterbar` 之间插入 `<section class="panel core-analysis">` 三卡  
   - 数据源：`extended` 的 `consistency` / `cognitive` / `recovery`（portrait.categories.context）  
   - 卡1 主值 = 一致性系数 `(1 - score/100)`；卡2 = 认知负荷 score；卡3 score=0/null 显示「未检出应激事件」  
   - 迷你曲线复用 `spark()` + `markerCurve`（能量 / 眨眼率）；关键发现从 cards/alerts 关键词匹配  
   - 八大章节（summary→data）未改动；`api.js` 空 portrait 回退补 `id`/`score`  
   - `python3 build.py` 已内嵌；服务已重启  
   - 自测：`report.html?task=13eb0d0bfe58` → 1.00 / 41.6 / 未检出应激事件；`?task=785e9a1781de` → 1.00 / 52.6 / 24.6

1. **18 页原型替换 React UI**  
   - 源码落地 `apps/web/`（`api.js` / `system.js` / `core.js` / CSS + `build.py`）  
   - FastAPI 托管 `/` 与 `/{page}.html`；`scripts/run.sh` 自动 `build.py`  
   - 原 React 移至 `apps/web/_legacy_react/`（未删，便于回滚）

2. **前端 API 适配层**（`apps/web/api.js`）  
   - `GET /api/jobs` → 任务列表（状态：queued/running/done/failed → queued/processing/completed/failed）  
   - `GET /api/jobs/{id}` → portrait.json 映射 metrics / extended / topics / transcript / events / markerData / heatmap / alerts / coverage / versions  
   - `POST /api/jobs` 接入单视频 / 批量 / 现场采集上传  
   - 隐藏 partial / cancelled；9 阶段由后端 stage 粗粒度推导

3. **重点指标三卡**（画像总览首屏）  
   - 语音情绪唤醒度（f0 + energy）  
   - 语速-停顿节律（CPM + 停顿 + 填充词）  
   - 认知负荷指数（`context.cognitive`）  
   - 六大类命名对齐后端：心理健康信号 / 情绪波动 / 焦虑信号 / 反应控制信号 / 注意分散 / 场景补充信号；无数据标注「未提取」

4. **GET /api/health 扩展**  
   - `openface`：`docker_available()` + 镜像 `inspect`（实测 available=true, mode=docker）  
   - `models` / `versions` / `system`（psutil CPU/内存/磁盘）

5. **第二轮补充：报告页硬编码清理**  
   - 重点发现 / 关键发现 / 日志 / 三层对比 / 贡献权重 / 导出接真实 portrait  
   - 全局清理「演示」字样；`python3 build.py` 内嵌 18 页

6. **第三轮修复：功能校验 P0/P1/P2（本轮）**  
   - **P0-1** `core.js` 预警中心改用 `alerts`（不再用 `events`/`cards`）；表格列适配 id/level/time_label/text  
   - **P1-1** `system.js` `rawObject()`：duration/coverage/versions/task.completed；alerts 用 boot.alerts；去掉 rPPG 假缺失与 EV-002  
   - **P1-2** `reportsPage()`：版本用 versions.pipeline/indicators/rules/topics；时间/任务用真实 task  
   - **P1-3** 热力图详情：`heatmap.markers` + `portrait.metrics.mean`；无基线显示「该标志物暂无基线」  
   - **P1-4** 关键词定位：从 segments/cards/transcript 查找出现时间，找不到不 seek  
   - **P1-5** 扩展指标贡献：从 context subs[].evidence 取；无则「暂无证据」  
   - **P1-6 / P1-7** 语义词云词频统计；时间轴按 duration 均分 5 刻度  
   - **P1-8** 标志物采样时间/回看用 `sampleSeekSec()`（首卡/首预警或 0）  
   - **P1-9** 待复核数默认 0，按 cards/events.reviewed 本地计算  
   - **P2-1** 任务中心：processing/queued 每 5s 轮询 `/api/jobs` +「刷新任务」按钮  
   - **P2-2** 预警图表定位用 `alerts[0].start_ms`；无 alerts 不渲染定位按钮  
   - `python3 build.py` 已重新内嵌；服务已重启

## 验收自测（本机 127.0.0.1:8765）

| 项 | 结果 |
| --- | --- |
| `./scripts/run.sh` 启动 | ✅ |
| 18 页 HTML 可打开 | ✅ |
| `/api/jobs` 真实任务 | ✅（含 `13eb0d0bfe58` AU 完整） |
| `/api/health` openface/models/versions/system | ✅ |
| 第三轮 grep（1122 / 0.972 / 09:38:42 / 0086 / EV-002 / 压力:75 / data-seek="730" / reviewcount">2 / 72,17,171…） | ✅ 无命中 |
| `report.html?task=13eb0d0bfe58` / `profile.html` | ✅ 200；预警区块走 alerts；无假 seek/复核数 |
| 有 alerts 的任务（如 `2b371baeb3ca`） | ✅ portrait.alerts 可加载 |
| 报告页三卡 `?task=13eb0d0bfe58` | ✅ HTTP 200；源码含 core-analysis；期望值 1.00 / 41.6 / 未检出应激事件 |
| 报告页三卡 `?task=785e9a1781de` | ✅ HTTP 200；期望值 1.00 / 52.6 / 24.6；八章节 id 仍在 |

## 遗留问题

1. **复核持久化**：按决策本版不做 SQLite review 字段；仅浏览器 localStorage。  
2. **rPPG / 生理**：v1 不做，标志物「生理」页标注未提取。  
3. **MediaPipe 探测**：venv 为 mp 0.10（无 classic solutions API）；health 以 `import mediapipe` 为就绪信号。  
4. **阶段进度**：仍为 stage 粗粒度映射，非 9 步精确百分比（二期可选）。  
5. **主题段级认知负荷**：portrait 未提供 per-topic 认知负荷；指标页主题段比较如实显示「暂无段级指标」。  
6. **短任务无预警**：如 `13eb0d0bfe58`（60s）alerts/cards 为空属正常；预警 UI 显示空态。

## 下一步

1. 浏览器走通：任务中心轮询 → 画像 / 报告预警中心（选有 alerts 的任务）→ data 查看器 / 报告中心。  
2. 可选：在 `api.js` 的 `mapPortrait` 顶层显式返回 `cards` / `deepseek`；清理 `_legacy_react/`、补 E2E 冒烟脚本。
