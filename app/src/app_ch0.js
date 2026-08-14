import { loadArtifact, showFatal } from "./model.js?v=2";
import { drawArch } from "./arch.js?v=2";

const $ = (s) => document.querySelector(s);
const art = await loadArtifact("data/artifact_ch0.json").catch((e) => { showFatal(e); throw e; });
const D = art.distributed, NC = art.nonconvex;

/* ---------- stats ---------- */
$("#d-active").textContent = Math.round(D.pct_active) + "%";
$("#d-pr").textContent = "~" + Math.round(D.participation_ratio);
$("#d-total").textContent = D.n_params.toLocaleString();
$("#nc-seeds").textContent = NC.n_seeds;
$("#nc-dist").textContent = NC.pair_dist_mean.toFixed(0);
$("#nc-norm").textContent = "≈" + NC.theta_norm.toFixed(0);
$("#nc-note").innerHTML = "every run: <b>retain 100%</b>, <b>p(Kyiv)=0</b> — identical behaviour, different weights.";

/* ---------- the transformer, top to bottom, each block tinted by the fact's footprint ---------- */
const mn = D.group_norms, maxN = Math.max(...Object.values(mn));
const ARCH = [
  { op: '<b>5 input tokens</b> — <span class="mono">the · capital · of · ukraine · is</span>' },
  { key: "token + position embeddings", shape: "each word → 32-dim vector", desc: "embed the word and its position" },
  { key: "block 1 · attention", shape: "2 heads · causal", desc: "each word gathers info from earlier words (self-attention)" },
  { key: "block 1 · MLP", shape: "32 → 64 → 32 · GELU", desc: "a per-word feed-forward transform" },
  { key: "block 2 · attention", shape: "2 heads · causal", desc: "a second round of attention" },
  { key: "block 2 · MLP", shape: "32 → 64 → 32 · GELU", desc: "a second feed-forward" },
  { key: "final layer-norm", shape: "+ tied output", desc: "normalize, then score all 30 words" },
  { op: '<b>softmax</b> → probability of the next word, e.g. <span class="mono">p(Kyiv)</span>' },
];
$("#arch").innerHTML = ARCH.map((s) => {
  if (s.op) return `<div class="arow op">↓&nbsp; ${s.op}</div>`;
  const n = mn[s.key], a = (0.05 + 0.18 * n / maxN).toFixed(3), w = (n / maxN * 100).toFixed(0);
  return `<div class="arow" style="background:rgba(124,92,255,${a})">
    <div><div class="aname">${s.key}</div><div class="ashape">${s.shape}</div></div>
    <div class="adesc">${s.desc}</div>
    <div class="aimp"><div class="atrack"><div class="abar" style="width:${w}%"></div></div><div class="aval">${n.toFixed(1)}</div></div>
  </div>`;
}).join("");

/* ============================================================
   parameter-space cloud (schematic, canvas, pseudo-3D)
   ============================================================ */
function mulberry32(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a);
  t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const rnd = mulberry32(7);
const randn = () => { let u = 0, v = 0; while (!u) u = rnd(); while (!v) v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };

const gray = [], factA = [], factB = [];
for (let i = 0; i < 150; i++) {
  let x = randn(), y = randn(), z = randn(); const r = Math.cbrt(rnd()) * 1.05 / Math.hypot(x, y, z);
  gray.push({ x: x * r, y: y * r, z: z * r });
}
const c0 = [-0.12, 0.06, 0.02];
for (let i = 0; i < 46; i++) {
  const a = { x: c0[0] + randn() * 0.32, y: c0[1] + randn() * 0.24, z: c0[2] + randn() * 0.28 };
  factA.push(a);
  factB.push({ x: c0[0] + 0.42 + (a.x - c0[0]) * 1.55, y: c0[1] - 0.06 + (a.y - c0[1]) * 1.4, z: c0[2] + 0.12 + (a.z - c0[2]) * 1.5 });
}

const cv = $("#cloud"), ctx = cv.getContext("2d");
let w = 0, h = 0, scale = 0, cx = 0, cy = 0;
function resize() {
  const dpr = window.devicePixelRatio || 1;
  w = cv.clientWidth; h = cv.clientHeight;
  cv.width = w * dpr; cv.height = h * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  scale = Math.min(w, h) * 0.34; cx = w / 2; cy = h / 2;
}
const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
const PUR = [124, 92, 255], GRN = [31, 184, 122];
function proj(p, ang) {
  const ca = Math.cos(ang), sa = Math.sin(ang);
  const xr = p.x * ca + p.z * sa, zr = -p.x * sa + p.z * ca;
  return { sx: cx + xr * scale, sy: cy - p.y * scale * 0.8 + zr * scale * 0.18, d: zr };
}
function axis(ang, from, to, label) {
  const a = proj({ x: from[0], y: from[1], z: from[2] }, ang), b = proj({ x: to[0], y: to[1], z: to[2] }, ang);
  ctx.strokeStyle = "#d3d8e4"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(a.sx, a.sy); ctx.lineTo(b.sx, b.sy); ctx.stroke();
  ctx.fillStyle = "#9aa3b2"; ctx.font = "italic 12px monospace"; ctx.fillText(label, b.sx + 3, b.sy + 3);
}
function draw(ang, m) {
  ctx.clearRect(0, 0, w, h);
  axis(ang, [-1.2, -1.05, -1.2], [0.1, -1.05, -1.2], "θ₃");
  axis(ang, [-1.2, -1.05, -1.2], [-1.2, 0.35, -1.2], "θ₂");
  axis(ang, [-1.2, -1.05, -1.2], [-1.2, -1.05, 0.2], "θ₁");
  const items = [];
  for (const p of gray) items.push({ p: proj(p, ang), col: [184, 191, 204], base: 1.7, fact: false });
  for (let i = 0; i < factA.length; i++) {
    const a = factA[i], b = factB[i];
    items.push({ p: proj({ x: a.x + (b.x - a.x) * m, y: a.y + (b.y - a.y) * m, z: a.z + (b.z - a.z) * m }, ang),
      col: mix(PUR, GRN, m), base: 3.1, fact: true });
  }
  items.sort((u, v) => u.p.d - v.p.d);
  for (const it of items) {
    const r = it.base * (0.7 + 0.4 * (it.p.d + 1.4) / 2.8);
    ctx.globalAlpha = it.fact ? 0.9 : 0.32 + 0.28 * (it.p.d + 1) / 2;
    ctx.fillStyle = `rgb(${it.col[0]},${it.col[1]},${it.col[2]})`;
    ctx.beginPath(); ctx.arc(it.p.sx, it.p.sy, r, 0, 7); ctx.fill();
  }
  ctx.globalAlpha = 1;
  if (m > 0.12) {                                   // green "removal" arrows radiating from the fact
    const cB = proj({ x: c0[0] + 0.42, y: c0[1] - 0.06, z: c0[2] + 0.12 }, ang);
    ctx.strokeStyle = `rgba(31,184,122,${(m * 0.85).toFixed(2)})`; ctx.fillStyle = ctx.strokeStyle; ctx.lineWidth = 2;
    for (let k = 0; k < 5; k++) {
      const a = k * (Math.PI * 2 / 5) + ang * 0.5, L = 34 * m;
      const ex = cB.sx + Math.cos(a) * L, ey = cB.sy + Math.sin(a) * L;
      ctx.beginPath(); ctx.moveTo(cB.sx, cB.sy); ctx.lineTo(ex, ey); ctx.stroke();
      ctx.beginPath(); ctx.arc(ex, ey, 2.4, 0, 7); ctx.fill();
    }
  }
}
resize(); window.addEventListener("resize", resize);
const reduce = window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches;
const slider = $("#removal");
let userM = null;                                   // null = auto-demo; set once the user drags
if (slider) slider.addEventListener("input", (e) => { userM = parseFloat(e.target.value); if (reduce) draw(0.6, userM); });
if (reduce) {
  draw(0.6, 0);
} else {
  let t0 = null;
  (function frame(ts) {
    if (t0 === null) t0 = ts; const e = (ts - t0) / 1000;
    const autoM = 0.5 - 0.5 * Math.cos((e % 6) / 6 * Math.PI * 2);   // 0→1→0 loop
    const m = userM !== null ? userM : autoM;
    if (slider && userM === null) slider.value = m;                 // slider tracks the demo until grabbed
    draw(e * 0.28, m);
    requestAnimationFrame(frame);
  })(performance.now());
}

/* ============================================================
   schematic non-convex landscape
   ============================================================ */
(function landscape() {
  const W = 560, H = 200, PAD = 22, N = 120;
  const G = (x, c, wd) => Math.exp(-(((x - c) / wd) ** 2));
  const f = (x) => -0.9 * G(x, 0.2, 0.09) - 1.0 * G(x, 0.46, 0.10) - 0.72 * G(x, 0.72, 0.085)
    - 0.86 * G(x, 0.9, 0.07) - 0.15 * Math.cos(6 * x);
  const xs = Array.from({ length: N }, (_, i) => i / (N - 1)), ys = xs.map(f);
  const lo = Math.min(...ys), hi = Math.max(...ys);
  const gx = (x) => PAD + x * (W - 2 * PAD);
  // low loss (lo) -> bottom, high loss (hi) -> top, so minima are valleys and dots sit in them
  const gy = (v) => H - PAD - (v - lo) / (hi - lo) * (H - 2 * PAD);
  const pts = xs.map((x, i) => `${gx(x).toFixed(1)},${gy(ys[i]).toFixed(1)}`).join(" ");
  const area = `M ${gx(0)},${H - PAD} ` + xs.map((x, i) => `L ${gx(x).toFixed(1)},${gy(ys[i]).toFixed(1)}`).join(" ") + ` L ${gx(1)},${H - PAD} Z`;
  const minima = [0.2, 0.46, 0.72, 0.9], labels = ["retrain A", "retrain B", "retrain C", "retrain D"];
  const balls = minima.map((x, i) =>
    `<circle cx="${gx(x).toFixed(1)}" cy="${(gy(f(x)) - 7).toFixed(1)}" r="5.5" fill="#1fb87a" stroke="#fff" stroke-width="2"/>
     <text x="${gx(x).toFixed(1)}" y="${(gy(f(x)) - 16).toFixed(1)}" text-anchor="middle" font-size="9.5" fill="#1fb87a" font-family="monospace">${labels[i]}</text>`).join("");
  $("#landscape").innerHTML = `
    <defs><linearGradient id="lg0" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#ef9a63"/><stop offset="0.55" stop-color="#f0d178"/><stop offset="1" stop-color="#8fb3ea"/></linearGradient></defs>
    <path d="${area}" fill="url(#lg0)" opacity="0.45"/>
    <polyline points="${pts}" fill="none" stroke="#7d5ba6" stroke-width="2"/>
    <text x="14" y="${H / 2}" font-size="11" fill="#5c6675" font-family="monospace" transform="rotate(-90 14 ${H / 2})">Loss L(θ)</text>
    <text x="${W - PAD}" y="${H - 6}" text-anchor="end" font-size="10" fill="#9aa3b2" font-family="monospace">parameters θ →</text>
    <text x="${PAD + 4}" y="${PAD + 2}" font-size="9.5" fill="#c2c8d4" font-family="monospace">schematic</text>
    ${balls}`;
})();

/* ============================================================
   real PCA scatter of the distinct retrained minima
   ============================================================ */
(function scatter() {
  const W = 420, H = 300, PAD = 26;
  const pts = NC.minima.map((m) => m.xy).concat([NC.full_xy]);
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const px = (x1 - x0) * 0.18 + 1, py = (y1 - y0) * 0.18 + 1; x0 -= px; x1 += px; y0 -= py; y1 += py;
  const gx = (x) => PAD + (x - x0) / (x1 - x0) * (W - 2 * PAD), gy = (y) => H - PAD - (y - y0) / (y1 - y0) * (H - 2 * PAD);
  let s = `<rect x="${PAD - 6}" y="${PAD - 6}" width="${W - 2 * PAD + 12}" height="${H - 2 * PAD + 12}" fill="none" stroke="#eef0f6" rx="10"/>
    <text x="${W / 2}" y="${H - 4}" text-anchor="middle" font-size="10" fill="#c2c8d4" font-family="monospace">PCA of the actual weight vectors</text>`;
  // faint lines between every pair of minima to show how far apart they are
  for (let i = 0; i < NC.minima.length; i++) for (let j = i + 1; j < NC.minima.length; j++)
    s += `<line x1="${gx(NC.minima[i].xy[0])}" y1="${gy(NC.minima[i].xy[1])}" x2="${gx(NC.minima[j].xy[0])}" y2="${gy(NC.minima[j].xy[1])}" stroke="#d8f0e5"/>`;
  for (const m of NC.minima)
    s += `<circle cx="${gx(m.xy[0]).toFixed(1)}" cy="${gy(m.xy[1]).toFixed(1)}" r="7" fill="#1fb87a" stroke="#fff" stroke-width="2"><title>seed ${m.seed}: retain ${(m.retain_acc * 100).toFixed(0)}%, p(kyiv)=${m.p_kyiv.toFixed(3)}</title></circle>`;
  s += `<circle cx="${gx(NC.full_xy[0]).toFixed(1)}" cy="${gy(NC.full_xy[1]).toFixed(1)}" r="7" fill="#1c2230" stroke="#fff" stroke-width="2"><title>θ_full · knows Kyiv</title></circle>`;
  $("#scatter").innerHTML = s;
})();

/* architecture diagram (no intervention yet — this is the model every chapter uses) */
if ($("#archdiag")) drawArch($("#archdiag"), { meta: art.meta, baked_info: null }, { mode: "none" });
