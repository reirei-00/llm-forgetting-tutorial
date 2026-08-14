import {
  loadArtifact, forwardTrace, topk, forgetTargetProb, retainAccuracy, showFatal,
} from "./model.js?v=2";
import { drawArch } from "./arch.js?v=4";

const $ = (s) => document.querySelector(s);
const pct = (x) => (x * 100).toFixed(x >= 0.995 ? 0 : 1) + "%";

const art = await loadArtifact("data/artifact_ch2.json").catch((e) => { showFatal(e); throw e; });
const F = art.prompts.forget;
const tok = (id) => art.vocab.tokens[id];
let variant = "plain", idx = 0;

const ckpt = () => art.variants[variant].checkpoints[idx].params;
const met = () => art.variants[variant].metrics[idx];

/* ---------- color helpers (shared idiom with Chapter 1) ---------- */
const LIGHT = [245, 247, 251], BLUE = [47, 123, 255], CORAL = [255, 92, 122];
const mix = (a, b, t) => `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(",")})`;
const actColor = (v, m) => { const t = m ? Math.max(-1, Math.min(1, v / m)) : 0;
  return t >= 0 ? mix(LIGHT, CORAL, t) : mix(LIGHT, BLUE, -t); };
const maxAbs = (arr) => arr.reduce((m, v) => Math.max(m, Math.abs(v)), 1e-9);

/* ---------- ticks (log-spaced GA steps) ---------- */
$("#ticks").innerHTML = art.steps.map((s, i) => `<span data-i="${i}">${s}</span>`).join("");
$("#ticks").querySelectorAll("span").forEach((el) =>
  el.addEventListener("click", () => setStep(parseInt(el.dataset.i))));

/* ---------- live transformer internals (embeddings · attention · output) ---------- */
const T = art.meta.seq_len, D = art.meta.d, HEADS = art.meta.heads, BLOCKS = art.meta.blocks;
const TOP = 78, eCellW = 20, eCellH = 7, colStep = 24, ex0 = 54;
const AG = 17, GA = 250, GB = GA + T * AG + 44;
const hidX = GB + T * AG + 34, hCellH = 7;
const outLabelX = hidX + 70, outBarX = outLabelX + 6, outBarW = 200;
const cell = (x, y, w, h, c) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${c}" rx="1.5"/>`;
const sLabel = (x, y, t) => `<text class="stage-label" x="${x}" y="${y}" text-anchor="middle">${t}</text>`;
const short = (s) => s.length > 5 ? s.slice(0, 4) + "." : s;
const attnColor = (w) => `rgb(${[245, 247, 251].map((c, i) => Math.round(c + ([124, 92, 255][i] - c) * w)).join(",")})`;
function attnGrid(p, x0, A, head) {
  p.push(sLabel(x0 + T * AG / 2, TOP - 30, `head ${head + 1}`));
  for (let k = 0; k < T; k++)
    p.push(`<text class="tok-label" x="${x0 + k * AG + AG / 2}" y="${TOP - 8}" text-anchor="start" font-size="9" transform="rotate(-45 ${x0 + k * AG + AG / 2} ${TOP - 8})">${short(tok(F.ids[k]))}</text>`);
  for (let i = 0; i < T; i++) {
    for (let k = 0; k < T; k++) p.push(cell(x0 + k * AG, TOP + i * AG, AG - 1, AG - 1, attnColor(A[i][k] || 0)));
    if (i === T - 1) p.push(`<rect x="${x0 - 1}" y="${TOP + i * AG - 1}" width="${T * AG + 1}" height="${AG + 1}" fill="none" stroke="#ff5c7a" stroke-width="1.5" rx="2"/>`);
  }
}
function drawNet(theta) {
  const tr = forwardTrace(art, theta, F.ids);
  const eMax = maxAbs(tr.emb.flatMap((r) => Array.from(r)));
  const last = tr.hidden[T - 1], hMax = maxAbs(Array.from(last));
  const p = [];
  p.push(sLabel(ex0 + (T * colStep - 6) / 2, 30, "token + position emb"));
  p.push(sLabel((GA + GB + T * AG) / 2, 30, `attention · block ${BLOCKS} (causal)`));
  p.push(sLabel(hidX + eCellW / 2, 30, "hidden @ “is”"));
  p.push(sLabel(outBarX + outBarW / 2 - 10, 30, "next-word probs"));
  for (let t = 0; t < T; t++) {
    const colX = ex0 + t * colStep;
    p.push(`<text class="tok-label" x="${colX + eCellW / 2}" y="${TOP - 8}" text-anchor="end" transform="rotate(-40 ${colX + eCellW / 2} ${TOP - 8})">${tok(F.ids[t])}</text>`);
    for (let j = 0; j < D; j++) p.push(cell(colX, TOP + j * eCellH, eCellW, eCellH - 1, actColor(tr.emb[t][j], eMax)));
  }
  const A = tr.attn[BLOCKS - 1];
  for (let i = 0; i < T; i++)
    p.push(`<text class="tok-label" x="${GA - 6}" y="${TOP + i * AG + AG - 4}" text-anchor="end" font-size="9" fill="${i === T - 1 ? "#ff5c7a" : "#5c6675"}">${short(tok(F.ids[i]))}</text>`);
  attnGrid(p, GA, A[0], 0); attnGrid(p, GB, A[1], 1);
  p.push(`<text class="tok-label" x="${(GA + GB + T * AG) / 2}" y="${TOP + T * AG + 16}" text-anchor="middle" fill="#ff5c7a">↑ what “is” attends to → the answer</text>`);
  for (let j = 0; j < D; j++) p.push(cell(hidX, TOP + j * hCellH, eCellW, hCellH - 1, actColor(last[j], hMax)));
  const order = Array.from(tr.probs.keys()).sort((a, b) => tr.probs[b] - tr.probs[a]).slice(0, 5);
  order.forEach((id, i) => {
    const y = TOP + i * 30, isT = id === F.answer_id, c = isT ? "#ff5c7a" : "#9aa3b2";
    p.push(`<text class="tok-label" x="${outLabelX}" y="${y + 13}" text-anchor="end" fill="${isT ? "#ff5c7a" : "#5c6675"}">${tok(id)}</text>`);
    p.push(`<rect x="${outBarX}" y="${y + 3}" width="${outBarW}" height="15" fill="#eef1f8" rx="4"/>`);
    p.push(`<rect x="${outBarX}" y="${y + 3}" width="${Math.max(1, tr.probs[id] * outBarW)}" height="15" fill="${c}" rx="4"/>`);
    p.push(`<text class="tok-label" x="${outBarX + outBarW + 6}" y="${y + 15}" fill="#5c6675">${pct(tr.probs[id])}</text>`);
  });
  $("#netflow").innerHTML = p.join("");
}

/* ---------- output bars ---------- */
function drawBars(theta) {
  const { top, other } = topk(art, theta, F.ids, 6);
  const rows = top.map((t) => {
    const cls = t.id === F.answer_id ? "prow target" : "prow";
    return `<div class="${cls}"><div class="tok">${t.token}</div>
      <div class="ptrack"><div class="pbar" style="width:${(t.p * 100).toFixed(1)}%"></div></div>
      <div class="pval">${pct(t.p)}</div></div>`;
  });
  rows.push(`<div class="prow other"><div class="tok">other vocab</div>
    <div class="ptrack"><div class="pbar" style="width:${(other * 100).toFixed(1)}%"></div></div>
    <div class="pval">${pct(other)}</div></div>`);
  $("#bars").innerHTML = rows.join("");
  $("#pred").textContent = top[0].token;
}

/* ---------- step curves (x = step index, evenly spaced, labeled by step) ---------- */
const CW = 560, CH = 240, PAD = 40, N = art.steps.length;
const sx = (i) => PAD + (i / (N - 1)) * (CW - 2 * PAD);
const sy = (v) => CH - PAD - v * (CH - 2 * PAD);
const polyI = (vals, color, dash = "") =>
  `<polyline points="${vals.map((v, i) => `${sx(i)},${sy(v)}`).join(" ")}" fill="none"
     stroke="${color}" stroke-width="2.5" stroke-dasharray="${dash}" stroke-linejoin="round" stroke-linecap="round"/>`;
function axesI(svgW) {
  let s = `<line x1="${PAD}" y1="${sy(0)}" x2="${CW - PAD}" y2="${sy(0)}" stroke="#e7e9f0"/>
    <line x1="${PAD}" y1="${sy(1)}" x2="${CW - PAD}" y2="${sy(1)}" stroke="#f1f2f7"/>
    <text x="${PAD - 6}" y="${sy(1) + 4}" text-anchor="end" font-size="10" fill="#9aa3b2" font-family="monospace">1.0</text>
    <text x="${PAD - 6}" y="${sy(0) + 4}" text-anchor="end" font-size="10" fill="#9aa3b2" font-family="monospace">0.0</text>`;
  art.steps.forEach((st, i) => { s += `<text x="${sx(i)}" y="${CH - 12}" text-anchor="middle" font-size="10" fill="#9aa3b2" font-family="monospace">${st}</text>`; });
  s += `<text x="${CW / 2}" y="${CH - 1}" text-anchor="middle" font-size="10" fill="#c2c8d4" font-family="monospace">GA step</text>`;
  return s;
}
function drawStepCurve() {
  const m = art.variants[variant].metrics;
  const pT = m.map((d) => d.p_target), rA = m.map((d) => d.retain_acc);
  const marker = `<line x1="${sx(idx)}" y1="${PAD - 8}" x2="${sx(idx)}" y2="${CH - PAD}" stroke="#c9cede" stroke-dasharray="3 3"/>
    <circle cx="${sx(idx)}" cy="${sy(pT[idx])}" r="5" fill="#ff5c7a" stroke="#fff" stroke-width="2"/>
    <circle cx="${sx(idx)}" cy="${sy(rA[idx])}" r="5" fill="#1fb87a" stroke="#fff" stroke-width="2"/>`;
  $("#stepcurve").innerHTML = axesI() + polyI(pT, "#ff5c7a") + polyI(rA, "#1fb87a") + marker;
}
function drawCompare() {
  const plain = art.variants.plain.metrics.map((d) => d.retain_acc);
  const stab = art.variants.stabilized.metrics.map((d) => d.retain_acc);
  const marker = `<line x1="${sx(idx)}" y1="${PAD - 8}" x2="${sx(idx)}" y2="${CH - PAD}" stroke="#c9cede" stroke-dasharray="3 3"/>`;
  $("#compare").innerHTML = axesI() + polyI(plain, "#ff5c7a", "5 4") + polyI(stab, "#1fb87a") + marker;
}

/* ---------- master ---------- */
function render() {
  const theta = ckpt(), m = met();
  drawNet(theta); drawBars(theta); drawStepCurve(); drawCompare();
  $("#step-val").textContent = art.steps[idx];
  $("#c-step").textContent = "step " + art.steps[idx];
  $("#s-forget").textContent = pct(m.p_target);
  $("#s-retain").textContent = pct(m.retain_acc);
  $("#s-loss").textContent = m.target_nll.toFixed(2);
  $("#s-entropy").textContent = m.entropy.toFixed(2);

  let msg;
  if (idx === 0) msg = "The deployed model. Kyiv is the confident answer.";
  else if (m.p_target > 0.2) msg = "Climbing: Kyiv is losing ground.";
  else if (m.entropy > 2) msg = "Sweet spot — Kyiv is gone, and the model is now openly unsure (high entropy).";
  else if (m.retain_acc > 0.5) msg = variant === "stabilized"
    ? "Anchor holding: Kyiv suppressed, most capitals still intact." : "Kyiv gone, but retained capitals are starting to fail.";
  else msg = "Over-ascended: coherence and retained knowledge are collapsing.";
  $("#ga-msg").textContent = msg;

  $("#ticks").querySelectorAll("span").forEach((el, i) => el.classList.toggle("hot", i === idx));
  $("#variant").querySelectorAll("button").forEach((b) =>
    b.classList.toggle("on", b.dataset.v === variant));
}

function setStep(i) { idx = i; $("#step").value = i; render(); }
$("#step").addEventListener("input", (e) => setStep(parseInt(e.target.value)));
$("#variant").querySelectorAll("button").forEach((b) =>
  b.addEventListener("click", () => { variant = b.dataset.v; render(); }));

/* ---------- intro: the loss landscape L(θ), with GA climbing out of the minimum ---------- */
function setupAscent() {
  const SL = art.loss_slice, dense = SL.dense, N = dense.length;
  const s_lo = SL.axis.s_lo, s_hi = SL.axis.s_hi, LMAX = 6.5;
  const W = 560, HT = 230, PADL = 50, PADR = 18, PADT = 18, PADB = 34;
  const gx = (s) => PADL + (s - s_lo) / (s_hi - s_lo) * (W - PADL - PADR);
  const gy = (L) => HT - PADB - Math.min(L, LMAX) / LMAX * (HT - PADB - PADT);
  const ds = (dense[N - 1].s - dense[0].s) / (N - 1);
  const at = (arr, s) => { let t = (s - dense[0].s) / ds; t = Math.max(0, Math.min(N - 1, t));
    const i = Math.floor(t), f = t - i; return dense[i][arr] + ((dense[Math.min(N - 1, i + 1)][arr]) - dense[i][arr]) * f; };
  const lerpL = (s) => at("Lf", s), lerpP = (s) => at("p", s), slope = (s) => (lerpL(s + ds) - lerpL(s - ds)) / (2 * ds);

  const curve = dense.map((d) => `${gx(d.s).toFixed(1)},${gy(d.Lf).toFixed(1)}`).join(" ");
  let area = `M ${gx(dense[0].s).toFixed(1)},${HT - PADB} `;
  dense.forEach((d) => (area += `L ${gx(d.s).toFixed(1)},${gy(d.Lf).toFixed(1)} `));
  area += `L ${gx(dense[N - 1].s).toFixed(1)},${HT - PADB} Z`;
  const tfx = gx(0), tfy = gy(lerpL(0));

  $("#ascent").innerHTML = `
    <defs><linearGradient id="lossgrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#ef9a63"/><stop offset="0.5" stop-color="#f0d178"/>
      <stop offset="1" stop-color="#8fb3ea"/></linearGradient></defs>
    <path d="${area}" fill="url(#lossgrad)" opacity="0.5"/>
    <polyline points="${curve}" fill="none" stroke="#7d5ba6" stroke-width="2"/>
    <line x1="${PADL}" y1="${PADT}" x2="${PADL}" y2="${HT - PADB}" stroke="#dfe3ec"/>
    <line x1="${PADL}" y1="${HT - PADB}" x2="${W - PADR}" y2="${HT - PADB}" stroke="#dfe3ec"/>
    <text x="18" y="${(HT - PADB + PADT) / 2}" font-size="11" fill="#5c6675" font-family="monospace" transform="rotate(-90 18 ${(HT - PADB + PADT) / 2})">Loss  L(θ)</text>
    <text x="${W - PADR}" y="${HT - 8}" text-anchor="end" font-size="10" fill="#9aa3b2" font-family="monospace">parameters θ  →</text>
    <circle cx="${tfx}" cy="${tfy}" r="4.5" fill="#1c2230"/>
    <text x="${tfx}" y="${tfy + 19}" text-anchor="middle" font-size="10.5" fill="#1c2230" font-family="monospace">θ_full · minimum</text>
    <text x="${tfx - 8}" y="${tfy - 11}" text-anchor="end" font-size="9.5" fill="#9aa3b2" font-family="monospace">training rolled ↓ in</text>
    <g id="ascent-dyn"></g>`;

  const dyn = $("#ascent-dyn");
  const paint = (s) => {
    const L = lerpL(s), x = gx(s), y = gy(L), sl = slope(s);
    const perS = (W - PADL - PADR) / (s_hi - s_lo), perL = -(HT - PADB - PADT) / LMAX;
    let vx = perS, vy = perL * sl; const m = Math.hypot(vx, vy) || 1, len = 42; vx = vx / m * len; vy = vy / m * len;
    const ux = vx / len, uy = vy / len, ax = x + vx, ay = y + vy;
    const bx = ax - ux * 8, by = ay - uy * 8;
    const trail = dense.filter((d) => d.s >= Math.min(0, s) && d.s <= Math.max(0, s))
      .map((d) => `${gx(d.s).toFixed(1)},${gy(d.Lf).toFixed(1)}`).join(" ");
    dyn.innerHTML =
      `<polyline points="${trail} ${x.toFixed(1)},${y.toFixed(1)}" fill="none" stroke="#ff5c7a" stroke-width="3" stroke-dasharray="2 4" stroke-linecap="round"/>` +
      `<line x1="${(x - vx * 0.45).toFixed(1)}" y1="${(y - vy * 0.45).toFixed(1)}" x2="${ax.toFixed(1)}" y2="${ay.toFixed(1)}" stroke="#ff5c7a" stroke-width="2"/>` +
      `<path d="M${ax.toFixed(1)},${ay.toFixed(1)} L${(bx - uy * 4).toFixed(1)},${(by + ux * 4).toFixed(1)} L${(bx + uy * 4).toFixed(1)},${(by - ux * 4).toFixed(1)} Z" fill="#ff5c7a"/>` +
      `<text x="${(ax + 4).toFixed(1)}" y="${(ay - 4).toFixed(1)}" font-size="10.5" fill="#ff5c7a" font-family="monospace">∇L_f</text>` +
      `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="7" fill="#ff5c7a" stroke="#fff" stroke-width="2"/>`;
    $("#ascent-cap").innerHTML =
      `climbing out of the minimum… <b>L(θ) = ${L.toFixed(2)}</b> · gradient slope = <b>${sl.toFixed(1)}</b> · p(Kyiv) = <b>${(lerpP(s) * 100).toFixed(0)}%</b>`;
  };
  if (window.matchMedia && matchMedia("(prefers-reduced-motion: reduce)").matches) { paint(s_hi * 0.75); return; }
  let start = null; const climbMs = 4400, cycle = climbMs + 1100;
  (function frame(ts) {
    if (start === null) start = ts;
    paint((s_hi * 0.92) * Math.min(1, ((ts - start) % cycle) / climbMs));
    requestAnimationFrame(frame);
  })(performance.now());
}
if ($("#archdiag")) drawArch($("#archdiag"), art, { mode: "grads" });
setupAscent();

/* ---------- honesty self-test: live inference == saved metric ---------- */
let maxDiff = 0;
for (const vk of ["plain", "stabilized"]) {
  const v = art.variants[vk];
  v.checkpoints.forEach((c, i) => {
    maxDiff = Math.max(maxDiff, Math.abs(forgetTargetProb(art, c.params) - v.metrics[i].p_target));
    maxDiff = Math.max(maxDiff, Math.abs(retainAccuracy(art, c.params) - v.metrics[i].retain_acc));
  });
}
const st = $("#selftest");
if (maxDiff < 3e-3) { st.classList.add("ok");
  st.textContent = `live inference matches the saved checkpoints (max Δ = ${maxDiff.toExponential(1)})`;
} else st.textContent = `⚠ live/reference mismatch: Δ = ${maxDiff.toExponential(2)}`;

render();
