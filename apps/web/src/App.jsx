import { useEffect, useMemo, useRef, useState } from "react";
import * as echarts from "echarts";

const TABS = [
  ["overview", "总览"],
  ["heatmap", "热力图"],
  ["cards", "特殊内容"],
  ["raw", "原始指标"],
];

const LEVEL = { alert: "关注", watch: "偏离", info: "关键词" };

export default function App() {
  const [jobs, setJobs] = useState([]);
  const [currentId, setCurrentId] = useState(null);
  const [job, setJob] = useState(null);
  const [file, setFile] = useState(null);
  const [narrate, setNarrate] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [tab, setTab] = useState("overview");
  const [group, setGroup] = useState("all");
  const videoRef = useRef(null);

  async function refreshList() {
    const response = await fetch("/api/jobs");
    if (!response.ok) return;
    setJobs(await response.json());
  }

  useEffect(() => {
    refreshList().catch((err) => setError(String(err)));
  }, []);

  useEffect(() => {
    if (!currentId) return undefined;
    let stop = false;
    async function pull() {
      const response = await fetch(`/api/jobs/${currentId}`);
      if (!response.ok || stop) return;
      const data = await response.json();
      setJob(data);
      if (data.status === "queued" || data.status === "running") {
        window.setTimeout(pull, 2000);
      } else {
        refreshList();
      }
    }
    pull();
    return () => {
      stop = true;
    };
  }, [currentId]);

  async function submit(event) {
    event.preventDefault();
    if (!file) return;
    setBusy(true);
    setError("");
    const body = new FormData();
    body.append("file", file);
    body.append("narrate", narrate ? "true" : "false");
    try {
      const response = await fetch("/api/jobs", { method: "POST", body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "上传失败");
      setCurrentId(data.id);
      setTab("overview");
      setJob(data);
      refreshList();
    } catch (err) {
      setError(String(err.message || err));
    } finally {
      setBusy(false);
    }
  }

  function seek(ms) {
    const node = videoRef.current;
    if (!node) return;
    node.currentTime = Math.max(0, ms / 1000);
    node.play().catch(() => {});
    node.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  const portrait = job?.portrait;
  const stageText = stageLabel(job);

  return (
    <div className="app">
      <header className="top">
        <div>
          <p className="mark">鸿心影鉴</p>
          <p className="sub">导入一段人物视频，对齐说话内容和身体反应，给出本次录像的参考指数、偏离和特殊内容。指数不是诊断。</p>
        </div>
      </header>
      <div className="layout">
        <aside>
          <form className="panel" onSubmit={submit}>
            <h2>导入</h2>
            <label className="file">
              {file ? file.name : "选择 mp4 / mov / mkv"}
              <input type="file" accept="video/mp4,video/quicktime,video/webm,.mkv" hidden onChange={(e) => setFile(e.target.files?.[0] || null)} />
            </label>
            <label className="check">
              <input type="checkbox" checked={narrate} onChange={(e) => setNarrate(e.target.checked)} />
              <span>使用 DeepSeek 生成叙述。开启后，转写文本和汇总数字会发到 DeepSeek，视频不会上传。</span>
            </label>
            <button className="primary" disabled={!file || busy}>{busy ? "正在提交" : "开始解析"}</button>
            {error ? <p className="error">{error}</p> : <p className="hint">单段超过 30 分钟会被拒绝。没有 Docker 时面部动作单元留空，其余结果仍会生成。</p>}
          </form>
          <div className="panel" style={{ marginTop: 16 }}>
            <h2>任务</h2>
            {jobs.length === 0 ? <p className="hint">还没有任务</p> : jobs.map((item) => (
              <button key={item.id} className={item.id === currentId ? "job active" : "job"} onClick={() => { setCurrentId(item.id); setTab("overview"); }}>
                <strong>{item.filename}</strong>
                <span>{stageLabel(item)} · {item.created_at?.slice(0, 16).replace("T", " ")}</span>
              </button>
            ))}
          </div>
        </aside>
        <main>
          {!job ? <p className="empty">选择左侧任务，或导入一段新视频。</p> : null}
          {job && job.status !== "done" ? (
            <section className="panel">
              <h2>{job.filename}</h2>
              <p>{stageText}</p>
              {job.error ? <p className="error">{job.error}</p> : <p className="hint">提取面部、语音和转写需要几分钟，页面会自己刷新。</p>}
            </section>
          ) : null}
          {portrait ? (
            <>
              <div className="tabs">
                {TABS.map(([id, label]) => (
                  <button key={id} className={tab === id ? "on" : ""} onClick={() => setTab(id)}>{label}</button>
                ))}
                <a href={`/api/jobs/${job.id}/metrics.csv`}><button type="button">下载指标 CSV</button></a>
                <button type="button" onClick={() => downloadJson(job)}>下载 JSON</button>
              </div>
              {portrait.baseline_limited ? <div className="banner">这段视频短于 30 秒，已算出参考指数，突变预警按规则关闭。</div> : null}
              <video ref={videoRef} src={job.video_url} controls />
              {tab === "overview" ? <Overview portrait={portrait} /> : null}
              {tab === "heatmap" ? <Heatmap portrait={portrait} onSeek={seek} /> : null}
              {tab === "cards" ? <Cards portrait={portrait} onSeek={seek} /> : null}
              {tab === "raw" ? <Raw portrait={portrait} group={group} setGroup={setGroup} /> : null}
              <footer className="footer">
                <div>{portrait.disclaimer}</div>
                <div>
                  指标 {portrait.versions?.indicators} · 规则 {portrait.versions?.rules} · 网格 {portrait.versions?.grid_note}
                  {portrait.deepseek?.requested
                    ? ` · DeepSeek ${portrait.deepseek.used ? "已接收转写" : "未完成叙述"} ${portrait.deepseek.model || ""}`
                    : " · 本次未向 DeepSeek 发送数据"}
                </div>
              </footer>
            </>
          ) : null}
        </main>
      </div>
    </div>
  );
}

function stageLabel(job) {
  if (!job) return "";
  const map = { queued: "排队", extracting: "提取特征", scoring: "计算画像", done: "完成", failed: "失败", running: "分析中" };
  return map[job.stage] || map[job.status] || job.status;
}

function downloadJson(job) {
  const blob = new Blob([JSON.stringify(job.portrait, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${job.filename}.portrait.json`;
  link.click();
  URL.revokeObjectURL(url);
}

function Overview({ portrait }) {
  const ref = useRef(null);
  const lineRef = useRef(null);
  useEffect(() => {
    const chart = echarts.init(ref.current);
    const cats = portrait.categories || [];
    chart.setOption({
      tooltip: { trigger: "item" },
      radar: {
        indicator: cats.map((cat) => ({ name: cat.score == null ? `${cat.name}\n未形成` : cat.name, max: 100 })),
        radius: "62%",
        axisName: { color: "#1c1915", fontSize: 12 },
      },
      series: [{
        type: "radar",
        data: [{ value: cats.map((cat) => (cat.score == null ? null : cat.score)), name: "关注强度", areaStyle: { color: "rgba(140,58,47,0.18)" }, lineStyle: { color: "#8c3a2f" }, itemStyle: { color: "#8c3a2f" } }],
      }],
    });
    const onResize = () => chart.resize();
    window.addEventListener("resize", onResize);
    return () => { window.removeEventListener("resize", onResize); chart.dispose(); };
  }, [portrait]);

  useEffect(() => {
    const node = lineRef.current;
    if (!node) return undefined;
    const chart = echarts.init(node);
    const series = portrait.series || {};
    const t = (series.t || []).map((ms) => `${Math.floor(ms / 1000)}s`);
    chart.setOption({
      tooltip: { trigger: "axis" },
      legend: { data: ["视线水平", "基频", "能量"] },
      grid: { left: 48, right: 18, top: 32, bottom: 28 },
      xAxis: { type: "category", data: t, name: "时间" },
      yAxis: { type: "value", name: "原值" },
      series: [
        { name: "视线水平", type: "line", showSymbol: false, data: series.gaze_yaw || [] },
        { name: "基频", type: "line", showSymbol: false, data: series.f0_hz || [] },
        { name: "能量", type: "line", showSymbol: false, data: series.energy_db || [] },
      ],
    });
    const onResize = () => chart.resize();
    window.addEventListener("resize", onResize);
    return () => { window.removeEventListener("resize", onResize); chart.dispose(); };
  }, [portrait]);

  return (
    <section className="score-grid">
      <div className="radar" ref={ref} />
      <div>
        {(portrait.categories || []).map((cat) => <Category key={cat.id} cat={cat} />)}
        <div className="line-chart" ref={lineRef} />
      </div>
    </section>
  );
}

function Category({ cat }) {
  const ready = cat.subs.filter((item) => item.status !== "not_available");
  const missing = cat.subs.filter((item) => item.status === "not_available").map((item) => item.name);
  return (
    <div className="cat">
      <div className="cat-head">
        <div>{cat.name} <span className="cov">{cat.coverage}</span></div>
        <b>{cat.score == null ? "—" : cat.score}</b>
      </div>
      {ready.map((item) => (
        <div key={item.id}>
          <div className="sub"><span>{item.name}</span><span>{item.score == null ? item.status : item.score}</span></div>
          {item.note ? <div className="evidence">{item.note}</div> : null}
          {item.evidence?.length ? <div className="evidence">{item.evidence.map((ev) => `${ev.name} ${ev.mean}${ev.z_p90 != null ? `，高位偏离 ${ev.z_p90}σ` : ""}`).join("；")}</div> : null}
        </div>
      ))}
      {missing.length ? <div className="missing">未提取：{missing.join("、")}</div> : null}
    </div>
  );
}

function Heatmap({ portrait, onSeek }) {
  const ref = useRef(null);
  const heat = portrait.heatmap || { segments: [], markers: [], z: [] };
  useEffect(() => {
    const chart = echarts.init(ref.current);
    const data = [];
    heat.z.forEach((row, y) => row.forEach((value, x) => { if (value != null) data.push([x, y, value]); }));
    const max = Math.max(2, ...data.map((item) => Math.abs(item[2])));
    chart.setOption({
      tooltip: { formatter: (p) => `${heat.segments[p.value[1]]?.label || ""} · ${heat.markers[p.value[0]]?.name || ""}：${p.value[2]}σ` },
      grid: { left: 88, right: 20, top: 20, bottom: 70 },
      xAxis: { type: "category", data: heat.markers.map((item) => item.name), axisLabel: { rotate: 40 } },
      yAxis: { type: "category", data: heat.segments.map((item) => item.label) },
      visualMap: { min: -max, max, calculable: true, orient: "horizontal", left: "center", bottom: 0, inRange: { color: ["#2f5d50", "#f3efe6", "#8c3a2f"] } },
      series: [{ type: "heatmap", data, label: { show: data.length < 80, formatter: (p) => p.value[2] } }],
    });
    chart.on("click", (params) => {
      const seg = heat.segments[params.value?.[1]];
      if (seg) onSeek(seg.start_ms);
    });
    const onResize = () => chart.resize();
    window.addEventListener("resize", onResize);
    return () => { window.removeEventListener("resize", onResize); chart.dispose(); };
  }, [portrait, onSeek]);
  return (
    <section className="panel">
      <h2>主题段 × 标志物偏离</h2>
      <p className="hint">颜色是相对本人全片基线的 Z 分数。点击一格，视频跳到该主题段。</p>
      <div className="chart" ref={ref} />
      <div>
        {(portrait.segments || []).map((seg) => (
          <button key={seg.id} className="job" onClick={() => onSeek(seg.start_ms)}>
            <strong>{seg.time_label} · {seg.topic} · {seg.polarity}</strong>
            <span>{seg.text}</span>
          </button>
        ))}
      </div>
    </section>
  );
}

function Cards({ portrait, onSeek }) {
  const cards = portrait.cards || [];
  if (!cards.length) return <p className="empty">没有触发关键词，也没有越过预设阈值的突变。</p>;
  return (
    <section>
      {(portrait.alerts || []).length ? <div className="banner">{portrait.alerts.length} 条偏离已列入预警。</div> : null}
      {cards.map((card) => <Card key={card.id} card={card} onSeek={onSeek} />)}
    </section>
  );
}

function Card({ card, onSeek }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!ref.current || !card.curve) return undefined;
    const chart = echarts.init(ref.current);
    chart.setOption({
      grid: { left: 40, right: 12, top: 16, bottom: 24 },
      xAxis: { type: "category", data: (card.curve.t || []).map((ms) => `${Math.floor(ms / 1000)}s`) },
      yAxis: { type: "value", name: card.curve.code },
      series: [{ type: "line", showSymbol: false, data: card.curve.values || [], lineStyle: { color: "#8c3a2f" } }],
    });
    return () => chart.dispose();
  }, [card]);
  const narrative = card.narrative;
  return (
    <article className="card">
      <header>
        <span className={`level ${card.level}`}>{LEVEL[card.level] || card.level}</span>
        <strong>{card.time_label}</strong>
        <button className="ghost" onClick={() => onSeek(card.start_ms)}>回看</button>
      </header>
      <p className="quote">{card.topic ? `${card.topic} · ` : ""}{card.text || "（该时段没有转写）"}</p>
      {card.keywords?.length ? <p className="hint">触发词：{card.keywords.join("、")}</p> : null}
      <div className="layers">
        <div>
          <p>相对本人基线</p>
          {(card.layer1 || []).slice(0, 3).map((item) => <div key={item.code}>{item.name} {item.direction} {item.z}σ</div>)}
        </div>
        <div>
          <p>相对其他主题</p>
          {card.layer2 ? <div>{card.layer2.topic} 在 {card.layer2.of} 段中排第 {card.layer2.rank}，{card.layer2.marker} {card.layer2.z}σ</div> : <div>没有可比主题段</div>}
          {(card.layer3 || []).slice(0, 3).map((item) => <div key={item.time_ms}>另一次 {item.time_label} {item.z == null ? "" : `${item.z}σ`}</div>)}
        </div>
      </div>
      <div className="line-chart" ref={ref} />
      {narrative ? (
        <div className="narrative">
          <div><b>现象</b>{narrative.phenomenon}</div>
          <div><b>对比</b>{narrative.comparison}</div>
          <div><b>趋势</b>{narrative.trend}</div>
          <div><b>预测</b>{narrative.prediction}</div>
        </div>
      ) : <p className="hint">叙述{card.narrative_status || "未生成"}。分数和对比不依赖大模型。</p>}
    </article>
  );
}

function Raw({ portrait, group, setGroup }) {
  const groups = useMemo(() => ["all", ...new Set((portrait.metrics || []).map((item) => item.group).filter(Boolean))], [portrait]);
  const rows = (portrait.metrics || []).filter((item) => group === "all" || item.group === group);
  return (
    <section className="panel">
      <h2>基座指标</h2>
      <p className="hint">{portrait.asr_text || "没有转写文本"}</p>
      <div className="filters">
        {groups.map((item) => <button key={item} className={item === group ? "on" : ""} onClick={() => setGroup(item)}>{item}</button>)}
      </div>
      <table>
        <thead><tr><th>指标</th><th>状态</th><th>均值</th><th>标准差</th><th>最小</th><th>最大</th></tr></thead>
        <tbody>
          {rows.map((item) => (
            <tr key={item.metric_code}>
              <td>{item.metric_name}</td>
              <td>{item.status}</td>
              <td>{fmt(item.mean)}</td>
              <td>{fmt(item.std)}</td>
              <td>{fmt(item.min)}</td>
              <td>{fmt(item.max)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

function fmt(value) {
  if (value == null || Number.isNaN(value)) return "—";
  return Number(value).toFixed(2);
}
