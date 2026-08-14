import {
  loadArtifact, forwardTrace, steerSelfTest, showFatal,
} from "./model.js?v=2";
import { drawArch } from "./arch.js?v=4";

const $ = (s) => document.querySelector(s);
const pct = (x) => (x * 100).toFixed(x >= 0.995 ? 0 : 1) + "%";
const HOT = [0, 0.5, 1.0, 1.5, 2.0];

const art = await loadArtifact("data/artifact_ch3.json").catch((e) => { showFatal(e); throw e; });
const F = art.prompts.forget;
const tok = (id) => art.vocab.tokens[id];
const D = art.meta.d;

const thetaFull = {}, thetaBaked = {};
for (const k of art.param_keys) {
  thetaFull[k] = Float64Array.from(art.theta_full[k]);
  thetaBaked[k] = Float64Array.from(art.theta_baked[k]);
}
const rOf = (l) => art.steering[String(l)];

let mode = "steer", layer = art.meta.best_layer, beta = 0;

function currentTrace(ids) {
  if (mode === "baked") return forwardTrace(art, thetaBaked, ids);
  return forwardTrace(art, thetaFull, ids, { layer, beta, r: rOf(layer) });
}
const currentProbs = (ids) => currentTrace(ids).probs;
function retainAcc() {
  let ok = 0;
  for (const p of art.prompts.retain) {
    const pr = currentProbs(p.ids);
    let bi = 0, best = -1;
    for (let i = 0; i < pr.length; i++) if (pr[i] > best) { best = pr[i]; bi = i; }
    if (bi === p.answer_id) ok++;
  }
  return ok / art.prompts.retain.length;
}

$("#ticks").innerHTML = HOT.map((b) => `<span data-b="${b}">${b.toFixed(1)}</span>`).join("");
$("#ticks").querySelectorAll("span").forEach((el) =>
  el.addEventListener("click", () => setBeta(parseFloat(el.dataset.b))));

/* ---------- activation space (fixed PCA basis from the pipeline) ---------- */
const P = art.pca;
const projPoint = (h) => {
  const out = [0, 0];
  for (let c = 0; c < 2; c++) { let s = 0; for (let j = 0; j < D; j++) s += (h[j] - P.mean[j]) * P.basis[c][j]; out[c] = s; }
  return out;
};
const rProj = (l) => {
  const r = rOf(l), out = [0, 0];
  for (let c = 0; c < 2; c++) { let s = 0; for (let j = 0; j < D; j++) s += r[j] * P.basis[c][j]; out[c] = s; }
  return out;
};
const AW = 560, AH = 380, APAD = 44;
const allPts = [P.points.forget, ...P.points.retain, ...P.points.nonce];
let ax0 = Math.min(...allPts.map((p) => p[0])), ax1 = Math.max(...allPts.map((p) => p[0]));
let ay0 = Math.min(...allPts.map((p) => p[1])), ay1 = Math.max(...allPts.map((p) => p[1]));
const apx = (ax1 - ax0) * 0.22 + 1, apy = (ay1 - ay0) * 0.22 + 1;
ax0 -= apx; ax1 += apx; ay0 -= apy; ay1 += apy;
const gx = (x) => APAD + (x - ax0) / (ax1 - ax0) * (AW - 2 * APAD);
const gy = (y) => AH - APAD - (y - ay0) / (ay1 - ay0) * (AH - 2 * APAD);

function drawActSpace() {
  const p = [`<defs><marker id="ar3" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
    <path d="M0,0 L7,3 L0,6 Z" fill="#7c5cff"/></marker></defs>`];
  p.push(`<rect x="${APAD - 12}" y="${APAD - 16}" width="${AW - 2 * APAD + 24}" height="${AH - 2 * APAD + 30}" fill="none" stroke="#eef0f6" rx="12"/>`);
  const nx = P.points.nonce.reduce((s, q) => s + gx(q[0]), 0) / P.points.nonce.length;
  const ny = P.points.nonce.reduce((s, q) => s + gy(q[1]), 0) / P.points.nonce.length;
  p.push(`<circle cx="${nx}" cy="${ny}" r="54" fill="#2f7bff" opacity="0.07"/>`);
  p.push(`<text x="${nx}" y="${ny - 60}" text-anchor="middle" font-size="10.5" fill="#2f7bff" font-family="monospace">"I can't answer" region</text>`);
  P.points.retain.forEach((q, i) =>
    p.push(`<circle cx="${gx(q[0])}" cy="${gy(q[1])}" r="5.5" fill="#1fb87a" stroke="#fff" stroke-width="1.5"><title>${P.labels.retain[i]}</title></circle>`));
  P.points.nonce.forEach((q, i) =>
    p.push(`<circle cx="${gx(q[0])}" cy="${gy(q[1])}" r="5.5" fill="#2f7bff" stroke="#fff" stroke-width="1.5"><title>${P.labels.nonce[i]}</title></circle>`));
  const f = P.points.forget;
  p.push(`<circle cx="${gx(f[0])}" cy="${gy(f[1])}" r="6.5" fill="#ff5c7a" stroke="#fff" stroke-width="2"><title>ukraine (unsteered)</title></circle>`);
  p.push(`<text x="${gx(f[0]) + 10}" y="${gy(f[1]) + 4}" font-size="10.5" fill="#ff5c7a" font-family="monospace">ukraine</text>`);
  const rp = rProj(layer);
  p.push(`<line x1="${gx(f[0])}" y1="${gy(f[1])}" x2="${gx(f[0] + rp[0])}" y2="${gy(f[1] + rp[1])}"
     stroke="#7c5cff" stroke-width="2" stroke-dasharray="5 4" marker-end="url(#ar3)" opacity=".8"/>`);
  p.push(`<text x="${(gx(f[0]) + gx(f[0] + rp[0])) / 2 + 8}" y="${(gy(f[1]) + gy(f[1] + rp[1])) / 2}" font-size="11" fill="#7c5cff" font-family="monospace">r</text>`);
  if (mode === "steer") {
    const q = projPoint(currentTrace(F.ids).residLast[layer]);
    p.push(`<circle cx="${gx(q[0])}" cy="${gy(q[1])}" r="10" fill="none" stroke="#ff5c7a" stroke-width="2.5"/>`);
    p.push(`<circle cx="${gx(q[0])}" cy="${gy(q[1])}" r="3" fill="#ff5c7a"/>`);
  }
  $("#actspace").innerHTML = p.join("");
  $("#act-note").textContent = mode === "steer"
    ? `The ring is Ukraine's steered state at β=${beta.toFixed(2)} (block ${layer + 1}). At β≈1 it sits inside the blue region — and the output below follows it.`
    : `Baked mode: the redirection lives in block ${art.baked_info.layer + 1}'s down-projection matrix — no injection, the map itself changed.`;
}

/* ---------- output bars ---------- */
function drawBars() {
  const probs = currentProbs(F.ids);
  const idx = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]).slice(0, 6);
  const rows = idx.map((id) => {
    const cls = id === F.answer_id ? "prow target" : (id === F.unknown_id ? "prow unknown" : "prow");
    return `<div class="${cls}"><div class="tok">${tok(id)}</div>
      <div class="ptrack"><div class="pbar" style="width:${(probs[id] * 100).toFixed(1)}%"></div></div>
      <div class="pval">${pct(probs[id])}</div></div>`;
  });
  const other = Math.max(0, 1 - idx.reduce((s, id) => s + probs[id], 0));
  rows.push(`<div class="prow other"><div class="tok">other vocab</div>
    <div class="ptrack"><div class="pbar" style="width:${(other * 100).toFixed(1)}%"></div></div>
    <div class="pval">${pct(other)}</div></div>`);
  $("#bars").innerHTML = rows.join("");
  $("#pred").textContent = tok(idx[0]);
  $("#s-forget").textContent = pct(probs[F.answer_id]);
  $("#s-unknown").textContent = pct(probs[F.unknown_id]);
  $("#s-retain").textContent = pct(retainAcc());
}

/* ---------- trade-off curve ---------- */
const CW = 560, CH = 240, CPAD = 40, BMAX = 2.0;
const cx = (b) => CPAD + (b / BMAX) * (CW - 2 * CPAD);
const cy = (v) => CH - CPAD - v * (CH - 2 * CPAD);
const poly = (pts, color) =>
  `<polyline points="${pts.map(([b, v]) => `${cx(b)},${cy(v)}`).join(" ")}" fill="none"
     stroke="${color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>`;
function drawCurve() {
  const sweep = art.reference[String(layer)];
  let s = `<line x1="${CPAD}" y1="${cy(0)}" x2="${CW - CPAD}" y2="${cy(0)}" stroke="#e7e9f0"/>
    <line x1="${CPAD}" y1="${cy(1)}" x2="${CW - CPAD}" y2="${cy(1)}" stroke="#f1f2f7"/>
    <text x="${CPAD - 6}" y="${cy(1) + 4}" text-anchor="end" font-size="10" fill="#9aa3b2" font-family="monospace">1.0</text>
    <text x="${CPAD - 6}" y="${cy(0) + 4}" text-anchor="end" font-size="10" fill="#9aa3b2" font-family="monospace">0.0</text>
    <text x="${cx(0)}" y="${CH - 8}" text-anchor="middle" font-size="10" fill="#9aa3b2" font-family="monospace">β=0</text>
    <text x="${cx(BMAX)}" y="${CH - 8}" text-anchor="middle" font-size="10" fill="#9aa3b2" font-family="monospace">β=2</text>`;
  s += poly(sweep.map((d) => [d.beta, d.p_kyiv]), "#ff5c7a");
  s += poly(sweep.map((d) => [d.beta, d.p_unknown]), "#2f7bff");
  s += poly(sweep.map((d) => [d.beta, d.retain_acc]), "#1fb87a");
  if (mode === "steer") {
    const probs = currentProbs(F.ids);
    s += `<line x1="${cx(beta)}" y1="${CPAD - 8}" x2="${cx(beta)}" y2="${CH - CPAD}" stroke="#c9cede" stroke-dasharray="3 3"/>
      <circle cx="${cx(beta)}" cy="${cy(probs[F.answer_id])}" r="5" fill="#ff5c7a" stroke="#fff" stroke-width="2"/>
      <circle cx="${cx(beta)}" cy="${cy(probs[F.unknown_id])}" r="5" fill="#2f7bff" stroke="#fff" stroke-width="2"/>
      <circle cx="${cx(beta)}" cy="${cy(retainAcc())}" r="5" fill="#1fb87a" stroke="#fff" stroke-width="2"/>`;
  }
  $("#curve").innerHTML = s;
}

/* ---------- master ---------- */
function render() {
  drawArch($("#archdiag"), art, { mode, layer });
  drawActSpace(); drawBars(); drawCurve();
  $("#beta-val").textContent = beta.toFixed(2);
  $("#c-beta").textContent = mode === "baked" ? "baked weights" : "β = " + beta.toFixed(2);
  $("#steer-controls").style.opacity = mode === "baked" ? "0.35" : "1";
  $("#steer-controls").style.pointerEvents = mode === "baked" ? "none" : "auto";
  $("#baked-stats").style.display = mode === "baked" ? "flex" : "none";
  if (mode === "baked") {
    const b = art.baked_info;
    $("#b-kyiv").textContent = pct(b.baked_metrics.p_kyiv);
    $("#b-retain").textContent = pct(b.baked_metrics.retain_acc);
    $("#b-changed").textContent = ((b.changed_params / b.total_params) * 100).toFixed(0) + "%";
    $("#mode-note").innerHTML = `The redirection is now <b>in the weights</b> — one down-projection matrix
      (${b.changed_params.toLocaleString()} of ${b.total_params.toLocaleString()} params), solved in closed form
      with retained activations pinned. Retain accuracy: <b>${pct(b.baked_metrics.retain_acc)}</b>.`;
  } else {
    $("#mode-note").innerHTML = `Drag β. The same vector is added to <b>every</b> query — watch what that
      does to the retained capitals as β grows. Then slide β back to 0: Kyiv returns <em>exactly</em>.`;
  }
  const probs = currentProbs(F.ids);
  let msg;
  if (mode === "baked") msg = "One matrix changed; forget redirected, retain pinned.";
  else if (beta < 0.05) msg = "Untouched. Kyiv is the confident answer.";
  else if (probs[F.unknown_id] > 0.6 && retainAcc() > 0.7) msg = "Sweet spot: a coherent 'unknown', most capitals intact.";
  else if (probs[F.unknown_id] > 0.6 && retainAcc() > 0.3) msg = "Kyiv redirected — but the blanket vector is starting to hit neighbours.";
  else if (retainAcc() <= 0.3) msg = "Over-steered: every query is being pushed toward 'unknown'.";
  else msg = "The thought is drifting toward the 'can't answer' region.";
  $("#c-msg").textContent = msg;
  $("#ticks").querySelectorAll("span").forEach((el) =>
    el.classList.toggle("hot", mode === "steer" && Math.abs(parseFloat(el.dataset.b) - beta) < 0.08));
  $("#mode").querySelectorAll("button").forEach((b) => b.classList.toggle("on", b.dataset.m === mode));
  $("#layer").querySelectorAll("button").forEach((b) => b.classList.toggle("on", parseInt(b.dataset.l) === layer));
}

function setBeta(b) { beta = b; $("#beta").value = b; render(); }
$("#beta").addEventListener("input", (e) => { beta = parseFloat(e.target.value); render(); });
$("#mode").querySelectorAll("button").forEach((b) =>
  b.addEventListener("click", () => { mode = b.dataset.m; render(); }));
$("#layer").querySelectorAll("button").forEach((b) =>
  b.addEventListener("click", () => { layer = parseInt(b.dataset.l); render(); }));

const diff = steerSelfTest(art);
const st = $("#selftest");
if (diff < 1e-4) { st.classList.add("ok");
  st.textContent = `live steered inference matches the reference pipeline (max Δ = ${diff.toExponential(1)})`;
} else st.textContent = `⚠ live/reference mismatch: Δ = ${diff.toExponential(2)}`;

render();
