/**
 * PsyTrace API adapter — maps backend /api/jobs + portrait.json → UI models.
 * Loaded before system.js / core.js.
 */
window.PsyTraceAPI = (() => {
  'use strict';

  const STATUS_MAP = {
    queued: 'queued',
    running: 'processing',
    done: 'completed',
    failed: 'failed',
  };

  const STAGE_FROM_BACKEND = {
    queued: 0,
    extracting: 4,
    scoring: 8,
    done: 9,
    failed: 0,
  };

  const TOPIC_COLORS = {
    健康: '#518b87',
    家庭: '#497e9f',
    工作: '#747fc7',
    经济: '#8a7a5c',
    情感: '#a07090',
    法律: '#6a7080',
    创伤: '#8b6060',
    自我评价: '#8979ac',
    其他: '#566d8d',
  };

  const CAT_META = [
    { id: 'mental', icon: 'brain', color: '#91a5ff', short: '心理健康信号' },
    { id: 'emotion', icon: 'pulse', color: '#7acbdb', short: '情绪波动' },
    { id: 'anxiety', icon: 'scan', color: '#c1a1e8', short: '焦虑信号' },
    { id: 'control', icon: 'shield', color: '#8ca9f0', short: '反应控制信号' },
    { id: 'attention', icon: 'eye', color: '#72c9c3', short: '注意分散' },
    { id: 'context', icon: 'layers', color: '#b0a0ff', short: '场景补充信号' },
  ];

  const EXT_META = [
    { id: 'cognitive', icon: 'layers', name: '认知负荷' },
    { id: 'conceal', icon: 'link', name: '掩饰-泄露' },
    { id: 'recovery', icon: 'pulse', name: '应激恢复' },
    { id: 'avoidance', icon: 'scan', name: '话题回避' },
    { id: 'micro_density', icon: 'eye', name: '微表情密度' },
    { id: 'consistency', icon: 'target', name: '语义-声学一致性' },
  ];

  const MARKER_TABS = [
    { name: '基频 F0', codes: ['f0_hz'], unit: 'Hz', group: 'acoustic' },
    { name: '面部 / AU', codes: ['au_04', 'face_confidence'], unit: '', group: 'face_head' },
    { name: '生理信号', codes: [], unit: '', group: 'physio', unavailable: true },
    { name: '眨眼 / 眼动', codes: ['blink_rate', 'gaze_yaw'], unit: '', group: 'gaze' },
    { name: '头部姿态', codes: ['head_yaw', 'head_pitch'], unit: '°', group: 'face_head' },
    { name: '语义 / ASR', codes: ['speech_rate_cpm', 'asr_confidence'], unit: '', group: 'asr' },
    { name: '声学能量', codes: ['energy_db'], unit: 'dB', group: 'acoustic' },
  ];

  async function getJSON(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url} → ${res.status}`);
    return res.json();
  }

  function fmtMs(ms) {
    const s = Math.max(0, Math.floor(Number(ms || 0) / 1000));
    return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
  }

  function downsample(arr, n = 22) {
    if (!arr || !arr.length) return Array(n).fill(0);
    const vals = arr.filter((v) => v != null && Number.isFinite(Number(v))).map(Number);
    if (!vals.length) return Array(n).fill(0);
    if (vals.length <= n) return vals;
    const out = [];
    for (let i = 0; i < n; i++) {
      const idx = Math.floor((i * (vals.length - 1)) / (n - 1));
      out.push(vals[idx]);
    }
    return out;
  }

  function mean(arr) {
    const vals = (arr || []).filter((v) => v != null && Number.isFinite(Number(v))).map(Number);
    if (!vals.length) return null;
    return vals.reduce((a, b) => a + b, 0) / vals.length;
  }

  function std(arr) {
    const vals = (arr || []).filter((v) => v != null && Number.isFinite(Number(v))).map(Number);
    if (vals.length < 2) return 0;
    const m = mean(vals);
    return Math.sqrt(vals.reduce((s, v) => s + (v - m) ** 2, 0) / vals.length);
  }

  function metricByCode(portrait, code) {
    return (portrait.metrics || []).find((m) => m.metric_code === code);
  }

  function mapStatus(backend) {
    return STATUS_MAP[backend] || backend || 'queued';
  }

  function mapStage(job) {
    if (job.status === 'done') return 9;
    if (job.status === 'failed') return 0;
    return STAGE_FROM_BACKEND[job.stage] ?? 0;
  }

  function mapJobToTask(job) {
    const portrait = job.portrait || null;
    const durationSec = portrait?.duration_ms ? Math.round(portrait.duration_ms / 1000) : 0;
    const subject = (job.filename || '').replace(/\.[^.]+$/, '') || job.id;
    const created = (job.created_at || '').replace('T', ' ').replace('Z', '').slice(0, 16);
    return {
      id: job.id,
      name: job.filename || job.id,
      subject,
      file: job.filename || 'input.mp4',
      source: '单视频',
      duration: durationSec,
      created,
      status: mapStatus(job.status),
      stage: mapStage(job),
      review: false,
      demo: false,
      size: 0,
      error: job.error || null,
      video_url: job.video_url || `/api/jobs/${job.id}/video`,
      backendStatus: job.status,
      backendStage: job.stage,
      hasPortrait: Boolean(portrait),
      updated_at: job.updated_at,
    };
  }

  function vocalArousal(portrait) {
    const f0 = metricByCode(portrait, 'f0_hz');
    const energy = metricByCode(portrait, 'energy_db');
    const series = portrait.series || {};
    const f0Series = downsample(series.f0_hz);
    const eSeries = downsample(series.energy_db);
    const f0Mean = f0?.mean ?? mean(series.f0_hz);
    const f0Std = f0?.std ?? std(series.f0_hz);
    const eMean = energy?.mean ?? mean(series.energy_db);
    // Descriptive 0–100: higher f0 variance + higher energy → higher arousal
    let score = null;
    if (f0Mean != null && eMean != null) {
      const varPart = Math.min(40, (f0Std || 0) * 1.2);
      const energyPart = Math.min(40, Math.max(0, (eMean + 70) * 1.5));
      const levelPart = Math.min(20, Math.max(0, (f0Mean - 120) / 6));
      score = Math.round(Math.min(100, Math.max(0, varPart + energyPart + levelPart)) * 10) / 10;
    }
    return {
      id: 'vocal_arousal',
      name: '语音情绪唤醒度',
      short: 'Vocal Arousal',
      score,
      unit: '分',
      icon: 'pulse',
      color: '#7acbdb',
      available: score != null,
      desc: '基频 f0 均值/方差 + 短时能量 energy_db',
      evidence: [
        f0Mean != null ? `f0 ${f0Mean.toFixed(1)} Hz` : null,
        f0Std != null ? `σ ${Number(f0Std).toFixed(1)}` : null,
        eMean != null ? `能量 ${eMean.toFixed(1)} dB` : null,
      ].filter(Boolean),
      values: f0Series.length ? f0Series : eSeries,
      children: ['基频均值', '基频方差', '短时能量'],
      subvalues: [
        f0Mean != null ? `${f0Mean.toFixed(1)} Hz` : '—',
        f0Std != null ? `${Number(f0Std).toFixed(1)}` : '—',
        eMean != null ? `${eMean.toFixed(1)} dB` : '—',
      ],
      contrib: ['基频 f0', '短时能量', '频谱质心'],
    };
  }

  function speechRhythm(portrait) {
    const cpm = metricByCode(portrait, 'speech_rate_cpm');
    const fillers = metricByCode(portrait, 'filler_count');
    const utt = metricByCode(portrait, 'utterance_count');
    const nonSpeech = metricByCode(portrait, 'non_speech_duration_ms');
    const dur = portrait.duration_ms || 1;
    const pauseRatio = nonSpeech?.mean != null ? nonSpeech.mean / dur : null;
    const cpmVal = cpm?.mean;
    let score = null;
    if (cpmVal != null) {
      // Mid-range speech rate scores higher as "rhythm stability" descriptive index
      const rateScore = Math.min(100, Math.max(0, 100 - Math.abs(cpmVal - 220) / 3));
      const pausePenalty = pauseRatio != null ? Math.min(30, pauseRatio * 40) : 0;
      const fillerPenalty = fillers?.mean != null ? Math.min(15, fillers.mean * 2) : 0;
      score = Math.round(Math.min(100, Math.max(0, rateScore - pausePenalty - fillerPenalty)) * 10) / 10;
    }
    return {
      id: 'speech_rhythm',
      name: '语速-停顿节律',
      short: 'Speech Rhythm',
      score,
      unit: '分',
      icon: 'chart',
      color: '#91a5ff',
      available: score != null,
      desc: '语速 CPM + 停顿占比 + 填充词 + 语段数',
      evidence: [
        cpmVal != null ? `${cpmVal.toFixed(0)} CPM` : null,
        pauseRatio != null ? `停顿 ${(pauseRatio * 100).toFixed(1)}%` : null,
        fillers?.mean != null ? `填充词 ${fillers.mean}` : null,
        utt?.mean != null ? `语段 ${utt.mean}` : null,
      ].filter(Boolean),
      values: downsample(
        (portrait.series?.energy_db || []).map((v, i, a) => {
          // proxy rhythm curve from energy envelope
          return v == null ? 0 : Math.abs(Number(v) + 50);
        }),
      ),
      children: ['语速 CPM', '停顿占比', '填充词', '语段数'],
      subvalues: [
        cpmVal != null ? `${cpmVal.toFixed(0)}` : '—',
        pauseRatio != null ? `${(pauseRatio * 100).toFixed(1)}%` : '—',
        fillers?.mean != null ? String(fillers.mean) : '—',
        utt?.mean != null ? String(utt.mean) : '—',
      ],
      contrib: ['语速', '停顿比例', '填充词次数'],
    };
  }

  function cognitiveLoad(portrait) {
    const ctx = (portrait.categories || []).find((c) => c.id === 'context');
    const sub = (ctx?.subs || []).find((s) => s.id === 'cognitive');
    const blink = metricByCode(portrait, 'blink') || metricByCode(portrait, 'blink_rate');
    const series = portrait.series || {};
    const score = sub?.status === 'ok' && sub.score != null ? Number(sub.score) : null;
    return {
      id: 'cognitive_load',
      name: '认知负荷指数',
      short: 'Cognitive Load',
      score,
      unit: '分',
      icon: 'layers',
      color: '#b0a0ff',
      available: score != null,
      desc: '眨眼率 + 语速 + 停顿 + 视线（context.cognitive）',
      evidence: [
        score != null ? `认知负荷 ${score}` : null,
        blink?.mean != null ? `眨眼相关 ${blink.mean}` : null,
        sub?.note || null,
      ].filter(Boolean),
      values: downsample(series.blink_rate || series.gaze_yaw),
      children: ['认知负荷', '眨眼', '语速/停顿', '视线'],
      subvalues: [
        score != null ? String(score) : '—',
        mean(series.blink_rate) != null ? mean(series.blink_rate).toFixed(1) : '—',
        '见 ASR',
        mean(series.gaze_yaw) != null ? `${mean(series.gaze_yaw).toFixed(1)}°` : '—',
      ],
      contrib: ['眨眼率', '语速下降', '停顿升高', '视线漂移'],
    };
  }

  function mapCategories(portrait) {
    const cats = portrait.categories || [];
    const series = portrait.series || {};
    return CAT_META.map((meta, i) => {
      const cat = cats.find((c) => c.id === meta.id) || {};
      const score = cat.score != null ? Number(cat.score) : null;
      const available = score != null;
      const subs = cat.subs || [];
      const okSubs = subs.filter((s) => s.status === 'ok' && s.score != null);
      const children = (okSubs.length ? okSubs : subs).slice(0, 6).map((s) => s.name);
      const subvalues = (okSubs.length ? okSubs : subs)
        .slice(0, 6)
        .map((s) => (s.score != null ? String(s.score) : s.status === 'not_available' ? '未提取' : '—'));
      const contrib = [];
      for (const s of okSubs) {
        for (const e of s.evidence || []) {
          if (e.name && contrib.length < 3) contrib.push(e.name);
        }
      }
      let values = downsample(series.f0_hz || series.gaze_yaw);
      if (meta.id === 'attention') values = downsample(series.gaze_yaw || series.blink_rate);
      if (meta.id === 'emotion' || meta.id === 'control') values = downsample(series.au_04 || series.head_yaw);
      if (meta.id === 'context') values = downsample(series.blink_rate || series.f0_hz);
      return {
        id: meta.id,
        name: cat.name || meta.short,
        short: meta.short,
        score: available ? score : 0,
        baseline: available ? score : 0,
        delta: available ? '—' : '未提取',
        unit: '分',
        icon: meta.icon,
        color: meta.color,
        desc: available ? `${cat.coverage || ''} 子项有效` : '当前任务无足够数据',
        direction: available ? '相对本人基线的偏离描述' : '未提取',
        available,
        coverage: cat.coverage || '',
        children: children.length ? children : ['暂无子项'],
        subvalues: subvalues.length ? subvalues : ['—'],
        contrib: contrib.length ? contrib : ['—'],
        values,
        subs,
      };
    });
  }

  function mapExtended(portrait) {
    const ctx = (portrait.categories || []).find((c) => c.id === 'context');
    const subs = ctx?.subs || [];
    return EXT_META.map((meta, i) => {
      const sub = subs.find((s) => s.id === meta.id);
      const ok = sub && sub.status === 'ok' && sub.score != null;
      return {
        id: meta.id,
        name: sub?.name || meta.name,
        value: ok ? String(sub.score) : '—',
        unit: ok ? '分' : '',
        delta: ok ? '—' : '未提取',
        quality: ok ? 'ok' : '未提取',
        hint: ok ? sub.note || '场景补充信号' : sub?.note || '未提取',
        icon: meta.icon,
        idx: 5,
        available: Boolean(ok),
        score: ok ? Number(sub.score) : null,
      };
    });
  }

  function mapTopics(portrait) {
    const segs = portrait.segments || [];
    const durMs = portrait.duration_ms || 1;
    const byTopic = new Map();
    for (const s of segs) {
      const key = s.topic || '其他';
      if (!byTopic.has(key)) byTopic.set(key, { name: key, start: s.start_ms / 1000, end: s.end_ms / 1000, texts: [], polarities: [] });
      const g = byTopic.get(key);
      g.start = Math.min(g.start, s.start_ms / 1000);
      g.end = Math.max(g.end, s.end_ms / 1000);
      if (s.text) g.texts.push(s.text);
      if (s.polarity) g.polarities.push(s.polarity);
    }
    // Prefer chronological segments as topic bars
    return segs.slice(0, 40).map((s) => ({
      name: s.topic || '其他',
      start: Math.round(s.start_ms / 1000),
      end: Math.round(s.end_ms / 1000),
      color: TOPIC_COLORS[s.topic] || TOPIC_COLORS['其他'],
      key: (s.text || '').slice(0, 24),
      polarity: s.polarity || '—',
      count: 1,
      text: s.text || '',
      id: s.id,
    }));
  }

  function mapTranscript(portrait) {
    const segs = portrait.segments || [];
    if (segs.length) {
      return segs.slice(0, 80).map((s) => ({
        time: Math.round(s.start_ms / 1000),
        text: s.text || '',
        topic: s.topic || '其他',
      }));
    }
    const asr = portrait.asr_text || '';
    if (!asr) return [];
    const chunk = 40;
    const out = [];
    for (let i = 0; i < asr.length; i += chunk) {
      out.push({ time: Math.round((i / asr.length) * ((portrait.duration_ms || 0) / 1000)), text: asr.slice(i, i + chunk), topic: '其他' });
    }
    return out.slice(0, 40);
  }

  function mapEvents(portrait) {
    const cards = portrait.cards || [];
    const triggerMap = { keyword: '关键词触发', marker: '标志物异常', both: '组合触发' };
    return cards.map((c, i) => ({
      id: c.id || `EV-${String(i + 1).padStart(3, '0')}`,
      time: Math.round((c.start_ms || 0) / 1000),
      end: Math.round((c.end_ms || c.start_ms || 0) / 1000),
      topic: c.topic || '其他',
      title: (c.text || '').slice(0, 48) || c.time_label || c.id,
      type: triggerMap[c.trigger] || c.trigger || '组合触发',
      key: (c.keywords || []).join('、') || '—',
      marker: (c.layer1 || []).map((x) => x.name || x.code).filter(Boolean).slice(0, 2).join(' / ') || '多模态',
      z: (c.layer1 && c.layer1[0] && c.layer1[0].z != null) ? `${c.layer1[0].z > 0 ? '+' : ''}${Number(c.layer1[0].z).toFixed(1)}` : '—',
      before: '—',
      after: '—',
      duration: Math.max(1, Math.round(((c.end_ms || c.start_ms || 0) - (c.start_ms || 0)) / 1000)),
      reviewed: false,
      level: c.level,
      narrative: c.narrative,
      layer1: c.layer1,
      layer2: c.layer2,
      layer3: c.layer3,
      curve: c.curve,
      raw: c,
    }));
  }

  function mapAlerts(portrait) {
    return (portrait.alerts || []).map((a) => ({
      id: a.id,
      level: a.level,
      time_label: a.time_label,
      text: a.text,
      start_ms: a.start_ms,
      time: Math.round((a.start_ms || 0) / 1000),
    }));
  }

  function mapMarkerData(portrait) {
    const metrics = portrait.metrics || [];
    const series = portrait.series || {};
    return MARKER_TABS.map((tab) => {
      if (tab.unavailable) {
        return {
          name: tab.name,
          value: '—',
          unit: '',
          base: '—',
          z: '—',
          coverage: '未提取',
          other: 'v1 不做 rPPG / 心率 / HRV / 呼吸',
          available: false,
          values: Array(22).fill(0),
        };
      }
      const m = tab.codes.map((c) => metricByCode(portrait, c) || metrics.find((x) => x.metric_code === c)).find(Boolean);
      const code = tab.codes.find((c) => series[c]) || tab.codes[0];
      const vals = downsample(series[code]);
      return {
        name: m?.metric_name || tab.name,
        value: m?.mean != null ? String(Number(m.mean).toFixed(m.mean >= 10 ? 1 : 3)) : '—',
        unit: m?.unit || tab.unit,
        base: m?.mean != null ? String(Number(m.mean).toFixed(2)) : '—',
        z: '—',
        coverage: m?.status === 'ok' ? 'ok' : '未提取',
        other: metrics
          .filter((x) => x.group === (m?.group || tab.group))
          .map((x) => x.metric_name)
          .slice(0, 6)
          .join('、') || '—',
        available: m?.status === 'ok',
        values: vals,
        code,
      };
    });
  }

  function mapHeatmap(portrait) {
    return portrait.heatmap || { segments: [], markers: [], z: [] };
  }

  function mapLogData(portrait, job) {
    return {
      demo: false,
      task_id: job?.id,
      models: portrait?.versions || {},
      config: { baseline: 'within_task_valid_intervals', grid_ms: portrait?.versions?.grid_ms },
      stages: ['queued', 'extracting', 'scoring', 'done'],
      deepseek: portrait?.deepseek,
      coverage: portrait?.coverage,
      missing_intervals: [],
    };
  }

  function buildFocusCards(portrait) {
    return [vocalArousal(portrait), speechRhythm(portrait), cognitiveLoad(portrait)];
  }

  function mapPortrait(portrait, job) {
    if (!portrait) return null;
    const durationSec = Math.round((portrait.duration_ms || 0) / 1000);
    const metrics = mapCategories(portrait);
    const focus = buildFocusCards(portrait);
    const task = {
      id: job?.id || portrait.job_id,
      subject: (job?.filename || portrait.filename || '').replace(/\.[^.]+$/, '') || portrait.job_id,
      file: job?.filename || portrait.filename || 'input.mp4',
      duration: durationSec,
      created: (job?.created_at || '').replace('T', ' ').replace('Z', ''),
      completed: (job?.updated_at || '').replace('T', ' ').replace('Z', ''),
      scene: '视频分析',
      version: portrait.versions?.pipeline || '0.1.0',
      demo: false,
      video_url: job?.video_url || `/api/jobs/${job?.id || portrait.job_id}/video`,
    };
    const markerData = mapMarkerData(portrait);
    const baseMarker = markerData.map((m) => {
      const n = Number(m.value);
      return Number.isFinite(n) ? n : 0;
    });
    const sdMarker = baseMarker.map((v) => Math.max(0.01, Math.abs(v) * 0.15 || 0.1));
    return {
      portrait,
      task,
      metrics,
      focus,
      extended: mapExtended(portrait),
      topics: mapTopics(portrait),
      transcript: mapTranscript(portrait),
      events: mapEvents(portrait),
      alerts: mapAlerts(portrait),
      markerData,
      baseMarker,
      sdMarker,
      heatmap: mapHeatmap(portrait),
      logData: mapLogData(portrait, job),
      coverage: portrait.coverage || {},
      versions: portrait.versions || {},
      duration: durationSec,
      series: portrait.series || {},
    };
  }

  function markerCurveFrom(mapped, index) {
    const m = mapped.markerData[index];
    if (m?.values?.length) return m.values;
    return Array(22).fill(0);
  }

  async function listJobs() {
    const jobs = await getJSON('/api/jobs');
    return jobs.map(mapJobToTask);
  }

  async function getJob(id) {
    return getJSON(`/api/jobs/${id}`);
  }

  async function getHealth() {
    return getJSON('/api/health');
  }

  async function createJob(file, narrate = true) {
    const fd = new FormData();
    fd.append('file', file);
    fd.append('narrate', narrate ? 'true' : 'false');
    const res = await fetch('/api/jobs', { method: 'POST', body: fd });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || `上传失败 ${res.status}`);
    }
    return res.json();
  }

  async function boot() {
    let jobs = [];
    try {
      jobs = await listJobs();
    } catch (err) {
      console.warn('listJobs failed', err);
      jobs = [];
    }
    const params = new URLSearchParams(location.search);
    const prefer = params.get('task');
    const completed = jobs.filter((t) => t.status === 'completed' && t.hasPortrait);
    let selected =
      (prefer && jobs.find((t) => t.id === prefer)) ||
      completed[0] ||
      jobs[0] ||
      null;

    let mapped = null;
    let rawJob = null;
    if (selected) {
      try {
        rawJob = await getJob(selected.id);
        selected = mapJobToTask(rawJob);
        if (rawJob.portrait) mapped = mapPortrait(rawJob.portrait, rawJob);
      } catch (err) {
        console.warn('getJob failed', err);
      }
    }

    // Fallback empty shell so UI still boots
    if (!mapped) {
      mapped = {
        portrait: null,
        task: selected
          ? {
              id: selected.id,
              subject: selected.subject,
              file: selected.file,
              duration: selected.duration || 0,
              created: selected.created,
              completed: '',
              scene: '视频分析',
              version: '—',
              demo: false,
              video_url: selected.video_url,
            }
          : {
              id: '—',
              subject: '无任务',
              file: '—',
              duration: 0,
              created: '',
              completed: '',
              scene: '—',
              version: '—',
              demo: false,
            },
        metrics: CAT_META.map((m) => ({
          id: m.id,
          name: m.short,
          short: m.short,
          score: 0,
          baseline: 0,
          delta: '未提取',
          unit: '分',
          icon: m.icon,
          color: m.color,
          desc: '尚无画像',
          direction: '未提取',
          available: false,
          children: ['未提取'],
          subvalues: ['—'],
          contrib: ['—'],
          values: Array(22).fill(0),
        })),
        focus: [
          { id: 'vocal_arousal', name: '语音情绪唤醒度', short: 'Vocal Arousal', score: null, unit: '分', icon: 'pulse', color: '#7acbdb', available: false, desc: '', evidence: [], values: Array(22).fill(0), children: [], subvalues: [], contrib: [] },
          { id: 'speech_rhythm', name: '语速-停顿节律', short: 'Speech Rhythm', score: null, unit: '分', icon: 'chart', color: '#91a5ff', available: false, desc: '', evidence: [], values: Array(22).fill(0), children: [], subvalues: [], contrib: [] },
          { id: 'cognitive_load', name: '认知负荷指数', short: 'Cognitive Load', score: null, unit: '分', icon: 'layers', color: '#b0a0ff', available: false, desc: '', evidence: [], values: Array(22).fill(0), children: [], subvalues: [], contrib: [] },
        ],
        extended: EXT_META.map((m) => ({ id: m.id, name: m.name, value: '—', unit: '', delta: '未提取', quality: '未提取', hint: '未提取', icon: m.icon, idx: 5, available: false, score: null })),
        topics: [],
        transcript: [],
        events: [],
        alerts: [],
        markerData: MARKER_TABS.map((t) => ({
          name: t.name,
          value: '—',
          unit: '',
          base: '—',
          z: '—',
          coverage: '未提取',
          other: t.unavailable ? 'v1 未提取' : '—',
          available: false,
          values: Array(22).fill(0),
        })),
        baseMarker: Array(7).fill(0),
        sdMarker: Array(7).fill(1),
        heatmap: { segments: [], markers: [], z: [] },
        logData: { demo: false, task_id: selected?.id, models: {}, config: {}, stages: [] },
        coverage: {},
        versions: {},
        duration: selected?.duration || 0,
        series: {},
      };
    }

    return {
      jobs,
      selected,
      rawJob,
      ...mapped,
      markerCurve: (index) => markerCurveFrom(mapped, index),
    };
  }

  return {
    boot,
    listJobs,
    getJob,
    getHealth,
    createJob,
    mapJobToTask,
    mapPortrait,
    mapStatus,
    mapStage,
    fmtMs,
    STATUS_MAP,
    STAGE_FROM_BACKEND,
  };
})();
