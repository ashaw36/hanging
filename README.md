# 鸿心影鉴

本地网页。导入一段人物视频后，先用内置的特征底座提取面部、头部、注视、声学、转写和行为事件，再计算参考指数、主题偏离和特殊内容卡片。

心率、呼吸、瞳孔直径在这一版留空，不填 0。面部动作单元需要本机 OpenFace 或 Docker；没有时相关子项为未提取，其余结果照常生成。

叙述可选。勾选后，转写文本和汇总数字会发给 DeepSeek，视频文件不会上传。不勾选时仍然出指数、热力图和预警。

## 运行

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # 需要叙述时填 DEEPSEEK_API_KEY
./scripts/run.sh
```

浏览器打开 http://127.0.0.1:8765 。单段视频超过 30 分钟会拒绝进入画像。

环境变量：

| 变量 | 默认 | 作用 |
|------|------|------|
| `ASR_MODEL` | `small` | faster-whisper 型号 |
| `STRIDE` | `2` | 视觉抽帧步长 |
| `DEEPSEEK_API_KEY` | 空 | 没有密钥时跳过叙述 |
| `PORT` | `8765` | 网页端口 |

阈值和权重在 `config/rules-0.1.0.yaml`、`config/indicators-0.1.0.yaml`。改完重新分析即可，不必改代码。
