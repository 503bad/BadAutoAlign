// レンダラ本体。
// - トラック表示（ズーム/パン対応の仮想化描画）とGainNodeベースの再生
// - 残差主導の3状態マーカー表示:
//     グレー = 合っている（補正済み含む）… 何もしなくてよい
//     黄     = まだズレが残っている（残差ms付き）… ドラッグで修正 → 再補正
//     赤     = 自動では判断できない … 聴いて確認、必要ならマーカーを作って合わせる
//     青     = 手動アンカー
// - マーカーのドラッグ（スクラブ再生付き）と「再補正」

const $ = (sel) => document.querySelector(sel);
const status = (msg) => { $("#status").textContent = msg; };

const RESIDUAL_TH_MS = 40;   // これ以上残っていたら「まだズレ」
const HIT_PX = 7;            // マーカーの当たり判定幅

const COLORS = {
  ok: "rgba(150,155,165,0.55)",
  off: "#e6b455",
  unknown: "#e66a6a",
  manual: "#5aa9e6",
};

const audioCtx = new AudioContext();

// track name -> {path, buffer, pyramid, gain}
const tracks = { guide: null, vocal: null, corrected: null, inst: null };

let reportNotes = [];        // 直近レポートのnotes
let markers = [];            // 表示用 {index, srcS, dstS, state, residualMs, appliedMs}
const manualAnchors = new Map();  // note index -> {srcS, dstS}
let lastMode = "timing";

let playing = null;
let playhead = 0;

const view = { pps: 0, scrollT: 0, MAX_PPS: 2000 };

// ---------------------------------------------------------------- utils

function laneOf(name) {
  return document.querySelector(`.lane[data-track="${name}"]`);
}

function maxDuration() {
  let d = 0;
  for (const t of Object.values(tracks)) {
    if (t && t.buffer) d = Math.max(d, t.buffer.duration);
  }
  return d;
}

function viewportWidth() {
  return document.querySelector(".lane-body").clientWidth;
}

function fitPps() {
  const dur = maxDuration();
  return dur > 0 ? viewportWidth() / dur : 0;
}

function ensureView() {
  const fit = fitPps();
  if (fit <= 0) return;
  if (view.pps <= 0) view.pps = fit;
  view.pps = Math.min(Math.max(view.pps, fit), view.MAX_PPS);
  const maxScroll = Math.max(0, maxDuration() - viewportWidth() / view.pps);
  view.scrollT = Math.min(Math.max(view.scrollT, 0), maxScroll);
}

function xToTime(lane, clientX) {
  return view.scrollT + (clientX - lane.getBoundingClientRect().left) / view.pps;
}

// ---------------------------------------------------------------- ピーク（ピラミッド）

function buildPyramid(buffer) {
  const data = buffer.getChannelData(0);
  const levels = [];
  let bin = 16;
  let lo = new Float32Array(Math.ceil(data.length / bin));
  let hi = new Float32Array(lo.length);
  for (let i = 0; i < lo.length; i++) {
    let a = 0, b = 0;
    const s0 = i * bin, s1 = Math.min(data.length, s0 + bin);
    for (let s = s0; s < s1; s++) {
      const v = data[s];
      if (v < a) a = v;
      if (v > b) b = v;
    }
    lo[i] = a; hi[i] = b;
  }
  levels.push({ bin, lo, hi });
  while (bin < 8192) {
    const prev = levels[levels.length - 1];
    bin *= 4;
    const n = Math.ceil(prev.lo.length / 4);
    const l = new Float32Array(n), h = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let a = 0, b = 0;
      for (let j = i * 4; j < Math.min(prev.lo.length, i * 4 + 4); j++) {
        if (prev.lo[j] < a) a = prev.lo[j];
        if (prev.hi[j] > b) b = prev.hi[j];
      }
      l[i] = a; h[i] = b;
    }
    levels.push({ bin, lo: l, hi: h });
  }
  return { levels, sampleRate: buffer.sampleRate };
}

function peakAt(pyramid, t0, t1) {
  const spp = (t1 - t0) * pyramid.sampleRate;
  let level = pyramid.levels[0];
  for (const lv of pyramid.levels) {
    if (lv.bin <= Math.max(spp, 16)) level = lv;
    else break;
  }
  const i0 = Math.floor((t0 * pyramid.sampleRate) / level.bin);
  const i1 = Math.max(i0 + 1, Math.ceil((t1 * pyramid.sampleRate) / level.bin));
  let lo = 0, hi = 0;
  for (let i = Math.max(0, i0); i < Math.min(level.lo.length, i1); i++) {
    if (level.lo[i] < lo) lo = level.lo[i];
    if (level.hi[i] > hi) hi = level.hi[i];
  }
  return [lo, hi];
}

// ---------------------------------------------------------------- マーカー状態

function buildMarkers() {
  markers = reportNotes.map((n) => {
    const manual = manualAnchors.get(n.index);
    const srcS = manual ? manual.srcS : n.anchor_src_s;
    const dstS = manual ? manual.dstS
      : (n.anchor_dst_s != null ? n.anchor_dst_s : n.start_s);
    let state;
    if (manual || n.manual) state = "manual";
    else if (n.anchor_src_s == null) state = "unmatched";  // 赤（ガイド側のみ）
    else if (n.timing_residual_ms == null) state = "unknown";
    else if (Math.abs(n.timing_residual_ms) >= RESIDUAL_TH_MS) state = "off";
    else state = "ok";
    return {
      index: n.index,
      srcS,
      dstS,
      state,
      residualMs: n.timing_residual_ms,
      appliedMs: n.timing_applied_ms,
    };
  });
  $("#btn-recorrect").disabled = manualAnchors.size === 0;
}

function markerStyle(m) {
  if (m.state === "manual") return { color: COLORS.manual, width: 2, dash: [] };
  if (m.state === "off") return { color: COLORS.off, width: 2, dash: [] };
  if (m.state === "unmatched" || m.state === "unknown") {
    return { color: COLORS.unknown, width: 1.5, dash: [3, 3] };
  }
  return { color: COLORS.ok, width: 1, dash: [] };
}

function tooltipText(m) {
  const ms = (v) => `${v >= 0 ? "+" : ""}${Math.round(v)}ms`;
  if (m.state === "manual") return "手動アンカー（再補正で反映されます）";
  if (m.state === "off") {
    return `まだ ${ms(m.residualMs)} ズレています — ドラッグで合わせて「再補正」`;
  }
  if (m.state === "unmatched") {
    return "ガイドに相当する音が見つかりません — ダブルクリックでマーカーを作成できます";
  }
  if (m.state === "unknown") return "自動では判断できない区間です — 聴いて確認してください";
  if (m.appliedMs != null && Math.abs(m.appliedMs) >= 35) {
    return `${ms(m.appliedMs)} 動かして合わせ済みです`;
  }
  return "元々合っていたため触っていません";
}

// ---------------------------------------------------------------- 描画

function drawLane(name) {
  const t = tracks[name];
  const lane = laneOf(name);
  const canvas = lane.querySelector("canvas");
  const body = lane.querySelector(".lane-body");
  const dpr = window.devicePixelRatio || 1;
  const w = body.clientWidth;
  const h = body.clientHeight;
  if (canvas.width !== w * dpr) canvas.width = w * dpr;
  if (canvas.height !== h * dpr) canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  if (t && t.pyramid) {
    ctx.fillStyle = name === "corrected" ? "#7bd0a0" : "#9db8d8";
    const mid = h / 2;
    for (let x = 0; x < w; x++) {
      const t0 = view.scrollT + x / view.pps;
      const t1 = view.scrollT + (x + 1) / view.pps;
      if (t0 > t.buffer.duration) break;
      const [lo, hi] = peakAt(t.pyramid, t0, t1);
      const y0 = mid + lo * mid * 0.95;
      const y1 = mid + hi * mid * 0.95;
      ctx.fillRect(x, y1, 1, Math.max(1, y0 - y1));
    }
  }

  // マーカー
  for (const m of markers) {
    const tSec = name === "guide" ? m.dstS
      : (name === "vocal") ? m.srcS : null;
    if (tSec == null) continue;
    const x = (tSec - view.scrollT) * view.pps;
    if (x < -20 || x > w + 20) continue;
    const st = markerStyle(m);
    ctx.strokeStyle = st.color;
    ctx.lineWidth = st.width;
    ctx.setLineDash(st.dash);
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.lineWidth = 1;
    // 「まだズレ」は残差を数値表示（操作の道しるべ）
    if (name === "vocal" && m.state === "off" && m.residualMs != null) {
      ctx.fillStyle = COLORS.off;
      ctx.font = "10px sans-serif";
      const label = `${m.residualMs >= 0 ? "+" : ""}${Math.round(m.residualMs)}ms`;
      ctx.fillText(label, x + 3, 11);
    }
  }

  // プレイヘッド
  const px = (playhead - view.scrollT) * view.pps;
  if (px >= 0 && px <= w) {
    ctx.strokeStyle = "#ffffff";
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, h);
    ctx.stroke();
  }
}

// 分:秒:1/100秒（左が分）
function formatTime(t) {
  if (!(t >= 0)) t = 0;
  const p = (v) => String(v).padStart(2, "0");
  return `${p(Math.floor(t / 60))}:${p(Math.floor(t % 60))}:${p(Math.floor((t % 1) * 100))}`;
}

// 表示範囲を8分割したタイムスタンプルーラー
function drawTimeRuler(name) {
  const lane = laneOf(name);
  const wrap = lane.querySelector(".lane-time");
  const canvas = wrap.querySelector("canvas");
  const dpr = window.devicePixelRatio || 1;
  const w = wrap.clientWidth;
  const h = wrap.clientHeight;
  if (canvas.width !== w * dpr) canvas.width = w * dpr;
  if (canvas.height !== h * dpr) canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (view.pps <= 0) return;
  const spanS = w / view.pps;
  ctx.strokeStyle = "#3a3f4b";
  ctx.fillStyle = "#8a8f9a";
  ctx.font = "9px sans-serif";
  for (let i = 0; i < 8; i++) {
    const x = Math.round((i / 8) * w) + 0.5;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
    ctx.fillText(formatTime(view.scrollT + (i / 8) * spanS), x + 3.5, h - 4);
  }
}

function drawConnectors() {
  const svg = $("#connector");
  svg.innerHTML = "";
  const wrapRect = $("#connector-wrap").getBoundingClientRect();
  const gRect = laneOf("guide").querySelector(".lane-body").getBoundingClientRect();
  const vRect = laneOf("vocal").querySelector(".lane-body").getBoundingClientRect();
  const w = viewportWidth();
  for (const m of markers) {
    if (m.srcS == null || m.dstS == null) continue;
    const x1 = (m.dstS - view.scrollT) * view.pps;
    const x2 = (m.srcS - view.scrollT) * view.pps;
    if ((x1 < 0 && x2 < 0) || (x1 > w && x2 > w)) continue;
    const st = markerStyle(m);
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", x1 + (gRect.left - wrapRect.left));
    line.setAttribute("y1", gRect.bottom - wrapRect.top);
    line.setAttribute("x2", x2 + (vRect.left - wrapRect.left));
    line.setAttribute("y2", vRect.top - wrapRect.top);
    line.setAttribute("stroke", st.color);
    line.setAttribute("stroke-width", "1");
    line.setAttribute("opacity", m.state === "ok" ? "0.35" : "0.8");
    if (st.dash.length) line.setAttribute("stroke-dasharray", st.dash.join(","));
    svg.appendChild(line);
  }
}

function drawAll() {
  ensureView();
  for (const name of Object.keys(tracks)) {
    drawLane(name);
    drawTimeRuler(name);
  }
  drawConnectors();
}

// ---------------------------------------------------------------- ツールチップ

const tooltip = document.createElement("div");
tooltip.id = "tooltip";
document.body.appendChild(tooltip);

function showTooltip(clientX, clientY, text) {
  tooltip.textContent = text;
  tooltip.style.display = "block";
  tooltip.style.left = `${clientX + 12}px`;
  tooltip.style.top = `${clientY + 14}px`;
}

function hideTooltip() {
  tooltip.style.display = "none";
}

function hitMarker(laneName, clientX) {
  const lane = laneOf(laneName).querySelector(".lane-body");
  const tSec = xToTime(lane, clientX);
  let best = null, bestDx = HIT_PX / view.pps;
  for (const m of markers) {
    const mt = laneName === "guide" ? m.dstS : m.srcS;
    if (mt == null) continue;
    const dx = Math.abs(mt - tSec);
    if (dx < bestDx) { best = m; bestDx = dx; }
  }
  return best;
}

// ---------------------------------------------------------------- ズーム/パン

$("#lanes").addEventListener("wheel", (e) => {
  if (fitPps() <= 0) return;
  if (e.ctrlKey || e.altKey) {
    e.preventDefault();
    const lane = e.target.closest(".lane-body");
    const rect = (lane || document.querySelector(".lane-body")).getBoundingClientRect();
    const anchorX = e.clientX - rect.left;
    const anchorT = view.scrollT + anchorX / view.pps;
    view.pps *= Math.exp(-e.deltaY * 0.01);
    ensureView();
    view.scrollT = anchorT - anchorX / view.pps;
    drawAll();
  } else {
    const delta = (Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY);
    if (delta === 0) return;
    e.preventDefault();
    view.scrollT += delta / view.pps;
    drawAll();
  }
}, { passive: false });

// ---------------------------------------------------------------- D&D

function setupDrop(name) {
  if (name === "corrected") return;
  const body = laneOf(name).querySelector(".lane-body");
  body.addEventListener("dragover", (e) => {
    e.preventDefault();
    body.classList.add("drag-over");
  });
  body.addEventListener("dragleave", () => body.classList.remove("drag-over"));
  body.addEventListener("drop", async (e) => {
    e.preventDefault();
    body.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (!file) return;
    await loadTrack(name, window.api.pathForFile(file));
  });
}

async function loadTrack(name, path) {
  status(`読み込み中: ${path}`);
  const buf = await window.api.readFile(path);
  const audio = await audioCtx.decodeAudioData(buf);
  const gain = tracks[name]?.gain || audioCtx.createGain();
  gain.connect(audioCtx.destination);
  tracks[name] = { path, buffer: audio, pyramid: buildPyramid(audio), gain };
  applyGain(name);
  const lane = laneOf(name);
  lane.querySelector(".lane-body").classList.add("has-audio");
  lane.querySelector(".lane-file").textContent = path;
  lane.querySelector(".lane-close").hidden = false;
  drawAll();
  updateButtons();
  status("準備完了");
}

// ---------------------------------------------------------------- トラックを閉じる

function closeTrack(name) {
  const t = tracks[name];
  if (!t) return;
  stopPlayback();
  stopHoldPreview();
  try { t.gain.disconnect(); } catch {}
  tracks[name] = null;
  const lane = laneOf(name);
  lane.querySelector(".lane-body").classList.remove("has-audio");
  lane.querySelector(".lane-file").textContent = "";
  lane.querySelector(".lane-close").hidden = true;
  // ガイド/ボーカルが無くなるとマーカー・手動アンカーは無効になる
  if (name === "guide" || name === "vocal") {
    reportNotes = [];
    manualAnchors.clear();
    buildMarkers();
  }
  if (!Object.values(tracks).some((x) => x && x.buffer)) {
    view.pps = 0;
    view.scrollT = 0;
    playhead = 0;
  }
  drawAll();
  updateButtons();
  status(`閉じました: ${lane.querySelector(".lane-name span").textContent}`);
}

function closeAll() {
  const any = Object.values(tracks).some((t) => t && t.buffer);
  if (!any) return;
  for (const name of Object.keys(tracks)) closeTrack(name);
  status("全トラックを閉じました");
}

// ---------------------------------------------------------------- 補正実行

function updateButtons() {
  const ready = tracks.guide && tracks.vocal;
  for (const id of ["#btn-timing", "#btn-pitch", "#btn-both"]) {
    $(id).disabled = !ready;
  }
  const anyLoaded = Object.values(tracks).some((t) => t && t.buffer);
  $("#btn-play").disabled = !anyLoaded;
  $("#btn-close-all").disabled = !anyLoaded;
  $("#btn-recorrect").disabled = manualAnchors.size === 0 || !ready;
  $("#btn-save").disabled = !tracks.corrected;
}

async function saveCorrected() {
  if (!tracks.corrected) return;
  try {
    const res = await window.api.saveAs({
      sourcePath: tracks.corrected.path,
      vocalPath: tracks.vocal ? tracks.vocal.path : null,
    });
    if (res.saved) status(`保存しました: ${res.filePath}`);
  } catch (err) {
    status(`保存エラー: ${err.message}`);
  }
}

// ---------------------------------------------------------------- 進捗表示

function showProgress() {
  $("#progress-overlay").classList.add("show");
  $(".progress-track").classList.add("indeterminate");
  $(".progress-fill").style.width = "0%";
  $(".progress-detail").textContent = "解析中…";
}

function hideProgress() {
  $("#progress-overlay").classList.remove("show");
}

window.api.onProgress((p) => {
  if (!p) return;
  if (p.total) {
    $(".progress-track").classList.remove("indeterminate");
    const pct = Math.min(100, Math.round((p.done / p.total) * 100));
    $(".progress-fill").style.width = `${pct}%`;
    $(".progress-detail").textContent = `フレーズ ${p.done} / ${p.total}`;
  } else if (p.log && $(".progress-track").classList.contains("indeterminate")) {
    // フレーズ総数が判明する前はバックエンドのログをそのまま見せる
    $(".progress-detail").textContent = p.log;
  }
});

async function runProcess(mode) {
  lastMode = mode;
  const options = {
    detector: "pyin",
    pitch_strength: parseFloat($("#opt-strength").value),
    max_shift_ms: parseFloat($("#opt-maxshift").value),
    timing_only: mode === "timing",
    pitch_only: mode === "pitch",
  };
  const manual = [...manualAnchors.entries()].map(([index, a]) => ({
    note_index: index, src_s: a.srcS, dst_s: a.dstS,
  }));
  for (const id of ["#btn-timing", "#btn-pitch", "#btn-both", "#btn-recorrect"]) {
    $(id).disabled = true;
  }
  status("補正処理中…");
  showProgress();
  try {
    const { report, output } = await window.api.process({
      input: tracks.vocal.path,
      guide: tracks.guide.path,
      options,
      manual_anchors: manual,
    });
    reportNotes = report.notes || [];
    buildMarkers();
    await loadTrack("corrected", output);
    const off = markers.filter((m) => m.state === "off").length;
    const unknown = markers.filter(
      (m) => m.state === "unmatched" || m.state === "unknown").length;
    status(`補正完了: 要確認 ${off + unknown}箇所（黄 ${off} / 赤 ${unknown}）`);
  } catch (err) {
    status(`エラー: ${err.message}`);
  } finally {
    hideProgress();
    updateButtons();
  }
}

// ---------------------------------------------------------------- ロータリーノブ

const SVG_NS = "http://www.w3.org/2000/svg";

function polar(cx, cy, r, deg) {
  const rad = ((deg - 90) * Math.PI) / 180;
  return [cx + r * Math.cos(rad), cy + r * Math.sin(rad)];
}

function createKnob(el, { value = 1, min = 0, max = 1.2, reset = 1, onChange }) {
  const A0 = -135, A1 = 135;
  const cx = 22, cy = 22, r = 15;
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("width", "44");
  svg.setAttribute("height", "52");

  const ring = document.createElementNS(SVG_NS, "path");
  ring.setAttribute("stroke", "#3a3f4b");
  ring.setAttribute("stroke-width", "3");
  ring.setAttribute("fill", "none");
  const arc = document.createElementNS(SVG_NS, "path");
  arc.setAttribute("stroke", "#5aa9e6");
  arc.setAttribute("stroke-width", "3");
  arc.setAttribute("fill", "none");
  const needle = document.createElementNS(SVG_NS, "line");
  needle.setAttribute("stroke", "#d7dae0");
  needle.setAttribute("stroke-width", "2");
  const label = document.createElementNS(SVG_NS, "text");
  label.setAttribute("x", "22");
  label.setAttribute("y", "50");
  label.setAttribute("text-anchor", "middle");
  svg.append(ring, arc, needle, label);
  el.appendChild(svg);

  function arcPath(a0, a1) {
    const [x0, y0] = polar(cx, cy, r, a0);
    const [x1, y1] = polar(cx, cy, r, a1);
    const large = a1 - a0 > 180 ? 1 : 0;
    return `M ${x0} ${y0} A ${r} ${r} 0 ${large} 1 ${x1} ${y1}`;
  }
  ring.setAttribute("d", arcPath(A0, A1));

  const knob = {
    value,
    set(v, fire = true) {
      knob.value = Math.min(max, Math.max(min, v));
      const a = A0 + ((knob.value - min) / (max - min)) * (A1 - A0);
      arc.setAttribute("d", knob.value > min ? arcPath(A0, a) : "");
      const [nx, ny] = polar(cx, cy, r - 4, a);
      needle.setAttribute("x1", cx);
      needle.setAttribute("y1", cy);
      needle.setAttribute("x2", nx);
      needle.setAttribute("y2", ny);
      label.textContent = `${Math.round(knob.value * 100)}%`;
      if (fire && onChange) onChange(knob.value);
    },
  };

  let drag = null;
  el.addEventListener("pointerdown", (e) => {
    drag = { y: e.clientY, v: knob.value };
    el.setPointerCapture(e.pointerId);
  });
  el.addEventListener("pointermove", (e) => {
    if (!drag) return;
    knob.set(drag.v + ((drag.y - e.clientY) / 150) * (max - min));
  });
  el.addEventListener("pointerup", () => { drag = null; });
  el.addEventListener("dblclick", () => knob.set(reset));
  el.addEventListener("wheel", (e) => {
    e.preventDefault();
    e.stopPropagation();
    knob.set(knob.value - Math.sign(e.deltaY) * 0.02);
  }, { passive: false });

  knob.set(value, false);
  return knob;
}

const knobs = {};

// ---------------------------------------------------------------- 再生（Gain常設）

function applyGain(name) {
  const t = tracks[name];
  if (!t || !t.gain) return;
  const muted = laneOf(name).querySelector(".lane-mute").classList.contains("active");
  const vol = knobs[name] ? knobs[name].value : 1;
  t.gain.gain.value = muted ? 0 : vol * vol;
}

function stopPlayback() {
  if (!playing) return;
  for (const s of playing.sources) {
    try { s.stop(); } catch {}
  }
  playhead = playing.offset + (audioCtx.currentTime - playing.startedAt);
  playing = null;
  $("#btn-play").textContent = "▶ 再生";
}

function startPlayback() {
  const sources = [];
  for (const t of Object.values(tracks)) {
    if (!t || !t.buffer) continue;
    const src = audioCtx.createBufferSource();
    src.buffer = t.buffer;
    src.connect(t.gain);
    src.start(0, playhead);
    sources.push(src);
  }
  if (!sources.length) return;
  playing = { sources, startedAt: audioCtx.currentTime, offset: playhead };
  $("#btn-play").textContent = "⏸ 停止";
  requestAnimationFrame(tick);
}

function tick() {
  if (!playing) { drawAll(); return; }
  playhead = playing.offset + (audioCtx.currentTime - playing.startedAt);
  if (playhead >= maxDuration()) { stopPlayback(); playhead = 0; }
  drawAll();
  requestAnimationFrame(tick);
}

// ---------------------------------------------------------------- スクラブ再生

let lastScrubAt = 0;

function scrubAt(timeS) {
  const t = tracks.vocal;
  if (!t || !t.buffer) return;
  const lenS = Math.min(0.2, Math.max(0.04,
    parseFloat($("#opt-scrub").value) / 1000 || 0.08));
  const now = audioCtx.currentTime;
  if (now - lastScrubAt < lenS * 0.75) return;  // 鳴りっぱなし防止
  lastScrubAt = now;
  const src = audioCtx.createBufferSource();
  src.buffer = t.buffer;
  const g = audioCtx.createGain();
  const fade = 0.005;
  const knob = knobs.vocal;
  const vol = knob ? knob.value * knob.value : 1;  // 音量ノブを適用
  g.gain.setValueAtTime(0, now);
  g.gain.linearRampToValueAtTime(vol, now + fade);
  g.gain.setValueAtTime(vol, now + lenS - fade);
  g.gain.linearRampToValueAtTime(0, now + lenS);
  src.connect(g).connect(audioCtx.destination);
  src.start(now, Math.max(0, timeS - lenS / 2), lenS);
}

// ---------------------------------------------------------------- 長押し試聴

let holdPreview = null;    // {src, gain}
let suppressNextClick = false;

// 指定トラックをその位置から、押している間だけ再生する。
// 音量ノブは適用する。ミュートは無視（押している＝聴きたい意思表示のため）
function startHoldPreview(trackName, fromS) {
  stopHoldPreview();
  const t = tracks[trackName];
  if (!t || !t.buffer) return;
  const src = audioCtx.createBufferSource();
  src.buffer = t.buffer;
  const g = audioCtx.createGain();
  const now = audioCtx.currentTime;
  const knob = knobs[trackName];
  const vol = knob ? knob.value * knob.value : 1;  // applyGainと同じカーブ
  g.gain.setValueAtTime(0, now);
  g.gain.linearRampToValueAtTime(vol, now + 0.005);
  src.connect(g).connect(audioCtx.destination);
  src.start(now, Math.max(0, fromS));
  holdPreview = { src, gain: g };
}

function stopHoldPreview() {
  if (!holdPreview) return;
  const { src, gain } = holdPreview;
  holdPreview = null;
  const now = audioCtx.currentTime;
  try {
    gain.gain.setValueAtTime(gain.gain.value, now);
    gain.gain.linearRampToValueAtTime(0, now + 0.01);
    src.stop(now + 0.015);
  } catch {}
}

window.addEventListener("pointerup", stopHoldPreview);
window.addEventListener("pointercancel", stopHoldPreview);

// ---------------------------------------------------------------- マーカー操作

let markerDrag = null;  // {marker, startSrcS, moved}

for (const laneName of ["vocal", "guide"]) {
  const body = laneOf(laneName).querySelector(".lane-body");

  body.addEventListener("pointermove", (e) => {
    if (markerDrag && laneName === "vocal") {
      if (!markerDrag.moved && Math.abs(e.clientX - markerDrag.startX) <= 3) {
        return;  // 3px以内は長押し試聴のまま（微動でアンカーを作らない）
      }
      if (!markerDrag.moved) stopHoldPreview();  // ドラッグへ移行
      const t = xToTime(body, e.clientX);
      markerDrag.marker.srcS = t;
      markerDrag.moved = true;
      manualAnchors.set(markerDrag.marker.index,
        { srcS: t, dstS: markerDrag.marker.dstS });
      markerDrag.marker.state = "manual";
      const dMs = (t - markerDrag.startSrcS) * 1000;
      showTooltip(e.clientX, e.clientY,
        `${dMs >= 0 ? "+" : ""}${Math.round(dMs)}ms（離すと確定 → 再補正）`);
      scrubAt(t);
      drawAll();
      return;
    }
    const m = hitMarker(laneName, e.clientX);
    if (m) {
      let text = tooltipText(m);
      if (laneName === "guide" && m.dstS != null) {
        text += " ／ 長押しでこの位置から試聴";
      } else if (laneName === "vocal" && m.srcS != null) {
        text += " ／ 長押しで試聴・ドラッグで移動";
      }
      showTooltip(e.clientX, e.clientY, text);
      body.style.cursor = (laneName === "vocal" && m.srcS != null)
        ? "ew-resize" : "default";
    } else {
      hideTooltip();
      body.style.cursor = "default";
    }
  });

  body.addEventListener("pointerleave", () => {
    hideTooltip();
    body.style.cursor = "default";
  });

  body.addEventListener("pointerdown", (e) => {
    if (laneName === "guide") {
      // ガイド側マーカーの長押し試聴（押している間だけ、ミュート無視）
      const m = hitMarker("guide", e.clientX);
      if (m && m.dstS != null) {
        startHoldPreview("guide", m.dstS);
        suppressNextClick = true;
        body.setPointerCapture(e.pointerId);
      }
      return;
    }
    const m = hitMarker("vocal", e.clientX);
    if (m && m.srcS != null) {
      // 押している間は試聴、3px以上動かしたらドラッグへ移行
      markerDrag = { marker: m, startSrcS: m.srcS, startX: e.clientX, moved: false };
      startHoldPreview("vocal", m.srcS);
      body.setPointerCapture(e.pointerId);
      e.stopPropagation();
    }
  });

  body.addEventListener("pointerup", () => {
    if (!markerDrag) return;
    if (markerDrag.moved) {
      status(`手動アンカーを設定しました（計 ${manualAnchors.size}個）— 「再補正」で反映されます`);
    }
    // 動かさず離した場合は長押し試聴だっただけなので何もしない
    markerDrag = null;
    hideTooltip();
    updateButtons();
    drawAll();
  });

  // ダブルクリック: ピン留め（vocal）/ 赤ノートへのマーカー作成（guide）
  body.addEventListener("dblclick", (e) => {
    const m = hitMarker(laneName, e.clientX);
    if (!m) return;
    if (laneName === "vocal" && m.srcS != null) {
      manualAnchors.set(m.index, { srcS: m.srcS, dstS: m.dstS });
      status("マーカーをピン留めしました（ガード無視で完全に合わせます）— 「再補正」で反映");
    } else if (laneName === "guide" && m.srcS == null) {
      manualAnchors.set(m.index, { srcS: m.dstS, dstS: m.dstS });
      status("ボーカル側にマーカーを作成しました — ドラッグで合わせる位置を指定してください");
    } else {
      return;
    }
    buildMarkers();
    updateButtons();
    drawAll();
  });
}

// クリックでプレイヘッド移動（マーカードラッグと衝突しないようclickで処理）
for (const lane of document.querySelectorAll(".lane-body")) {
  lane.addEventListener("click", (e) => {
    if (view.pps <= 0) return;
    if (suppressNextClick) {
      suppressNextClick = false;  // ガイド試聴の離し際にプレイヘッドを動かさない
      return;
    }
    if (hitMarker("vocal", e.clientX) && lane === laneOf("vocal").querySelector(".lane-body")) {
      return;  // マーカー上のクリックはプレイヘッドを動かさない
    }
    playhead = xToTime(lane, e.clientX);
    drawAll();
  });
}

// ---------------------------------------------------------------- init

for (const name of Object.keys(tracks)) {
  setupDrop(name);
  const lane = laneOf(name);
  knobs[name] = createKnob(lane.querySelector(".knob"), {
    value: 1, reset: 1, onChange: () => applyGain(name),
  });
  const muteBtn = lane.querySelector(".lane-mute");
  muteBtn.addEventListener("click", () => {
    muteBtn.classList.toggle("active");
    applyGain(name);
  });
  lane.querySelector(".lane-close").addEventListener("click", () => closeTrack(name));
}
$("#btn-close-all").addEventListener("click", closeAll);
$("#btn-timing").addEventListener("click", () => runProcess("timing"));
$("#btn-pitch").addEventListener("click", () => runProcess("pitch"));
$("#btn-both").addEventListener("click", () => runProcess("both"));
$("#btn-recorrect").addEventListener("click", () =>
  runProcess(lastMode === "pitch" ? "both" : lastMode));
$("#btn-save").addEventListener("click", saveCorrected);
$("#btn-play").addEventListener("click", () => (playing ? stopPlayback() : startPlayback()));
// スペースキーで再生/一時停止（入力欄フォーカス中は除く）
window.addEventListener("keydown", (e) => {
  if (e.code !== "Space" || e.repeat) return;
  if (/^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) return;
  e.preventDefault();  // フォーカス中ボタンの誤発火・スクロールを防ぐ
  if ($("#btn-play").disabled) return;
  if (playing) stopPlayback();
  else startPlayback();
});
window.addEventListener("resize", drawAll);

window.api.version()
  .then((v) => status(`バックエンド接続OK (vat ${v.version})`))
  .catch((e) => status(`バックエンド起動失敗: ${e.message} — uvとPython環境を確認してください`));
