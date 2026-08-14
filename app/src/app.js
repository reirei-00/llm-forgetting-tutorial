import {
  loadArtifact, thetaAt, topk, forwardTrace, projectAlpha,
  forgetTargetProb, retainAccuracy, selfTest, showFatal,
} from "./model.js?v=2";
import { drawArch } from "./arch.js?v=2";

const $ = (s) => document.querySelector(s);
const pct = (x) => (x * 100).toFixed(x >= 0.995 ? 0 : 1) + "%";
const HOT = [0.0, 0.5, 1.0, 1.25];
const SVGNS = "http://www.w3.org/2000/svg";

const art = await loadArtifact("data/artifact.json").catch((e) => { showFatal(e); throw e; });
const F = art.prompts.forget;
const tok = (id) => art.vocab.tokens[id];

/* ---------- color helpers ---------- */
const LIGHT = [245, 247, 251], BLUE = [47, 123, 255], CORAL = [255, 92, 122];
const mix = (a, b, t) => `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(",")})`;
const actColor = (v, m) => {
  const t = m ? Math.max(-1, Math.min(1, v / m)) : 0;
  return t >= 0 ? mix(LIGHT, CORAL, t) : mix(LIGHT, BLUE, -t);
};
const maxAbs = (arr) => arr.reduce((m, v) => Math.max(m, Math.abs(v)), 1e-9);

/* ---------- static: lineage + module norms + ticks ---------- */
$("#ln-full").textContent = "‖Δ‖ " + art.lineage.theta0_to_full_norm.toFixed(1);
$("#ln-fgt").textContent = "‖v_f‖ " + art.lineage.vf_norm.toFixed(1);

// aggregate the ~34 transformer tensors into readable groups (combined L2 norm)
const norms = art.lineage.vf_module_norms;
function groupOf(k) {
  if (k === "tok" || k === "pos") return "token + position embeddings";
  const m = k.match(/^b(\d+)\./);
  if (m) {
    const blk = +m[1] + 1;
    return /W[qkv]|b[qkv]|Wo|bo|ln1/.test(k) ? `block ${blk} · attention` : `block ${blk} · MLP`;
  }
  return "final layer-norm";
}
const groups = {};
for (const k of art.param_keys) { const g = groupOf(k); (groups[g] ??= []).push(norms[k]); }
const combined = Object.fromEntries(Object.entries(groups).map(([g, a]) =>
  [g, Math.sqrt(a.reduce((s, v) => s + v * v, 0))]));
const order = ["token + position embeddings", "block 1 · attention", "block 1 · MLP",
  "block 2 · attention", "block 2 · MLP", "final layer-norm"].filter((g) => g in combined);
const maxNorm = Math.max(...Object.values(combined));
$("#modnorms").innerHTML = order.map((g) =>
  `<div class="mod"><div class="lbl">${g}</div>
   <div class="track"><div class="bar" style="width:${(combined[g] / maxNorm * 100).toFixed(0)}%"></div></div>
   <div class="val">${combined[g].toFixed(1)}</div></div>`).join("");

$("#ticks").innerHTML = HOT.map((a) => `<span data-a="${a}">${a.toFixed(2)}</span>`).join("");
$("#ticks").querySelectorAll("span").forEach((el) =>
  el.addEventListener("click", () => setAlpha(parseFloat(el.dataset.a))));

/* =========================================================================
   PANEL A · weight-space vector diagram
   ========================================================================= */
const G = art.geometry;
const VW = 560, VH = 380;
const vpts = [G.theta0, G.theta_full, G.theta_reinf, G.theta_ref,
              projectAlpha(art, 0), projectAlpha(art, 1.25)];
let xmin = Math.min(...vpts.map((p) => p[0])), xmax = Math.max(...vpts.map((p) => p[0]));
let ymin = Math.min(...vpts.map((p) => p[1])), ymax = Math.max(...vpts.map((p) => p[1]));
const padX = (xmax - xmin) * 0.14 + 4, padY = (ymax - ymin) * 0.14 + 4;
xmin -= padX; xmax += padX; ymin -= padY; ymax += padY;
const PL = { x0: 46, x1: 524, y0: 26, y1: 344 };
const vx = (x) => PL.x0 + ((x - xmin) / (xmax - xmin)) * (PL.x1 - PL.x0);
const vy = (y) => PL.y1 - ((y - ymin) / (ymax - ymin)) * (PL.y1 - PL.y0);

const arrowDef = (id, color) =>
  `<marker id="${id}" markerWidth="9" markerHeight="9" refX="7" refY="3"
     orient="auto"><path d="M0,0 L7,3 L0,6 Z" fill="${color}"/></marker>`;
const vdot = (p, color, r, title) =>
  `<circle cx="${vx(p[0])}" cy="${vy(p[1])}" r="${r + 3.5}" fill="#fff"/>` +
  `<circle cx="${vx(p[0])}" cy="${vy(p[1])}" r="${r}" fill="${color}"><title>${title}</title></circle>`;
const vchip = (x, y, text, color, anchor = "middle") => {
  const w = text.length * 6.6 + 14, h = 18;
  const rx = anchor === "start" ? x : anchor === "end" ? x - w : x - w / 2;
  const tx = anchor === "start" ? x + 7 : anchor === "end" ? x - 7 : x;
  return `<rect x="${rx}" y="${y - h / 2}" width="${w}" height="${h}" rx="9" fill="#fff" stroke="${color}" stroke-opacity=".35"/>` +
    `<text x="${tx}" y="${y + 4}" text-anchor="${anchor}" class="vpoint-label" fill="${color}">${text}</text>`;
};

function drawVec(alpha) {
  const cur = projectAlpha(art, alpha);
  const distRef = Math.hypot(cur[0] - G.theta_ref[0], cur[1] - G.theta_ref[1]);
  const p = [`<defs>${arrowDef("aviolet", "#7c5cff")}${arrowDef("acoral", "#ff5c7a")}
    <filter id="vglow" x="-60%" y="-60%" width="220%" height="220%"><feGaussianBlur stdDeviation="4"/></filter></defs>`];

  // frame + faint dotted grid
  p.push(`<rect x="${PL.x0}" y="${PL.y0}" width="${PL.x1 - PL.x0}" height="${PL.y1 - PL.y0}" fill="none" stroke="#e9ebf3" rx="10"/>`);
  for (let i = 1; i < 5; i++) {
    p.push(`<line x1="${vx(xmin + (xmax - xmin) * i / 5)}" y1="${PL.y0}" x2="${vx(xmin + (xmax - xmin) * i / 5)}" y2="${PL.y1}" stroke="#f2f3f9"/>`);
    p.push(`<line x1="${PL.x0}" y1="${vy(ymin + (ymax - ymin) * i / 5)}" x2="${PL.x1}" y2="${vy(ymin + (ymax - ymin) * i / 5)}" stroke="#f2f3f9"/>`);
  }
  // zero axes
  p.push(`<line x1="${PL.x0}" y1="${vy(0)}" x2="${PL.x1}" y2="${vy(0)}" stroke="#dfe3ec"/>`);
  p.push(`<line x1="${vx(0)}" y1="${PL.y0}" x2="${vx(0)}" y2="${PL.y1}" stroke="#dfe3ec"/>`);
  p.push(`<text class="vlabel" x="${(PL.x0 + PL.x1) / 2}" y="${PL.y1 + 22}" text-anchor="middle" fill="#9aa3b2">${G.axis_labels[0]} →</text>`);
  p.push(`<text class="vlabel" x="${PL.x0 + 2}" y="${PL.y0 - 6}" fill="#9aa3b2">↑ ${G.axis_labels[1]}</text>`);

  // v_f arrow (theta_full -> theta_reinf: "drill Kyiv deeper") and the negation arrow
  p.push(`<line x1="${vx(G.theta_full[0])}" y1="${vy(G.theta_full[1])}" x2="${vx(G.theta_reinf[0])}" y2="${vy(G.theta_reinf[1])}"
     stroke="#7c5cff" stroke-width="2.5" marker-end="url(#aviolet)"/>`);
  if (alpha > 0.001)
    p.push(`<line x1="${vx(G.theta_full[0])}" y1="${vy(G.theta_full[1])}" x2="${vx(cur[0])}" y2="${vy(cur[1])}"
       stroke="#ff5c7a" stroke-width="3" marker-end="url(#acoral)"/>`);
  // dashed distance to the retrained ideal
  p.push(`<line x1="${vx(cur[0])}" y1="${vy(cur[1])}" x2="${vx(G.theta_ref[0])}" y2="${vy(G.theta_ref[1])}"
     stroke="#2f7bff" stroke-width="1.3" stroke-dasharray="4 4" opacity=".65"/>`);

  // points (halo + hover title). Identity lives in the legend; the plot stays uncluttered.
  p.push(vdot(G.theta0, "#9aa3b2", 5, "θ₀ · base"));
  p.push(vdot(G.theta_reinf, "#ff5c7a", 5, "θ_reinf · Kyiv drilled deeper"));
  p.push(vdot(G.theta_ref, "#2f7bff", 6, "θ₋f · retrained without Kyiv (the ideal)"));
  p.push(vdot(G.theta_full, "#1c2230", 6, "θ_full · the model we fix"));
  p.push(`<circle cx="${vx(cur[0])}" cy="${vy(cur[1])}" r="10" fill="#7c5cff" opacity=".28" filter="url(#vglow)"/>`);
  p.push(vdot(cur, "#7c5cff", 7, "θ(α) · current model"));

  // a few chips: the two arrows + the moving point (with a leader line)
  p.push(vchip((vx(G.theta_full[0]) + vx(G.theta_reinf[0])) / 2 - 16, (vy(G.theta_full[1]) + vy(G.theta_reinf[1])) / 2, "v_f", "#7c5cff"));
  if (alpha > 0.06)
    p.push(vchip((vx(G.theta_full[0]) + vx(cur[0])) / 2 + 8, (vy(G.theta_full[1]) + vy(cur[1])) / 2, `−${alpha.toFixed(2)}·v_f`, "#ff5c7a", "start"));
  const chipY = vy(cur[1]) - 26;
  p.push(`<line x1="${vx(cur[0])}" y1="${vy(cur[1]) - 9}" x2="${vx(cur[0])}" y2="${chipY + 9}" stroke="#c9c2f0"/>`);
  p.push(vchip(vx(cur[0]), chipY, "θ(α)", "#7c5cff"));

  $("#vecspace").innerHTML = p.join("");
  $("#vec-note").innerHTML =
    `θ(α) is <b>${distRef.toFixed(1)}</b> from the ideal θ₋f. It slides straight down −v_f — ` +
    `passing near, then overshooting, the reference. It never lands on it.`;
}

/* =========================================================================
   PANEL B · live transformer internals (embeddings · attention · output)
   ========================================================================= */
const T = art.meta.seq_len, D = art.meta.d, HEADS = art.meta.heads, BLOCKS = art.meta.blocks;
const TOP = 78;
const eCellW = 20, eCellH = 7, colStep = 24, ex0 = 54;       // embedding columns
const AG = 17, GA = 250, GB = GA + T * AG + 44;              // attention grids (2 heads)
const hidX = GB + T * AG + 34, hCellH = 7;                   // final hidden @ last token
const outLabelX = hidX + 70, outBarX = outLabelX + 6, outBarW = 200;
const cell = (x, y, w, h, c) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="${c}" rx="1.5"/>`;
const sLabel = (x, y, t) => `<text class="stage-label" x="${x}" y="${y}" text-anchor="middle">${t}</text>`;
const short = (s) => s.length > 5 ? s.slice(0, 4) + "." : s;
const attnColor = (w) => `rgb(${[245, 247, 251].map((c, i) => Math.round(c + ([124, 92, 255][i] - c) * w)).join(",")})`;

function attnGrid(p, x0, A, head) {
  p.push(sLabel(x0 + T * AG / 2, TOP - 30, `head ${head + 1}`));
  for (let k = 0; k < T; k++)                                  // key labels (columns)
    p.push(`<text class="tok-label" x="${x0 + k * AG + AG / 2}" y="${TOP - 8}" text-anchor="start" font-size="9"
      transform="rotate(-45 ${x0 + k * AG + AG / 2} ${TOP - 8})">${short(tok(F.ids[k]))}</text>`);
  for (let i = 0; i < T; i++) {
    for (let k = 0; k < T; k++)
      p.push(cell(x0 + k * AG, TOP + i * AG, AG - 1, AG - 1, attnColor(A[i][k] || 0)));
    if (i === T - 1)                                           // highlight the query row that predicts the answer
      p.push(`<rect x="${x0 - 1}" y="${TOP + i * AG - 1}" width="${T * AG + 1}" height="${AG + 1}" fill="none" stroke="#ff5c7a" stroke-width="1.5" rx="2"/>`);
  }
}

function drawNet(alpha) {
  const tr = forwardTrace(art, thetaAt(art, alpha), F.ids);
  const eMax = maxAbs(tr.emb.flatMap((r) => Array.from(r)));
  const last = tr.hidden[T - 1], hMax = maxAbs(Array.from(last));
  const p = [];

  p.push(sLabel(ex0 + (T * colStep - 6) / 2, 30, "token + position emb"));
  p.push(sLabel((GA + GB + T * AG) / 2, 30, `attention · block ${BLOCKS} (causal)`));
  p.push(sLabel(hidX + eCellW / 2, 30, "hidden @ “is”"));
  p.push(sLabel(outBarX + outBarW / 2 - 10, 30, "next-word probs"));

  // embeddings, one column per token
  for (let t = 0; t < T; t++) {
    const colX = ex0 + t * colStep;
    p.push(`<text class="tok-label" x="${colX + eCellW / 2}" y="${TOP - 8}" text-anchor="end"
      transform="rotate(-40 ${colX + eCellW / 2} ${TOP - 8})">${tok(F.ids[t])}</text>`);
    for (let j = 0; j < D; j++) p.push(cell(colX, TOP + j * eCellH, eCellW, eCellH - 1, actColor(tr.emb[t][j], eMax)));
  }

  // attention: last block, both heads. Row = query (who looks), col = key (looked at).
  const A = tr.attn[BLOCKS - 1];
  for (let i = 0; i < T; i++)                                  // query labels (rows), left of grid A
    p.push(`<text class="tok-label" x="${GA - 6}" y="${TOP + i * AG + G - 4}" text-anchor="end" font-size="9"
      fill="${i === T - 1 ? "#ff5c7a" : "#5c6675"}">${short(tok(F.ids[i]))}</text>`);
  attnGrid(p, GA, A[0], 0);
  attnGrid(p, GB, A[1], 1);
  p.push(`<text class="tok-label" x="${(GA + GB + T * AG) / 2}" y="${TOP + T * AG + 16}" text-anchor="middle" fill="#ff5c7a">↑ what “is” attends to → the answer</text>`);

  // final hidden at the last position
  for (let j = 0; j < D; j++) p.push(cell(hidX, TOP + j * hCellH, eCellW, hCellH - 1, actColor(last[j], hMax)));

  // output: top-5 next-word probabilities
  const idx = Array.from(tr.probs.keys()).sort((a, b) => tr.probs[b] - tr.probs[a]).slice(0, 5);
  idx.forEach((id, i) => {
    const y = TOP + i * 30, isT = id === F.answer_id, c = isT ? "#ff5c7a" : "#9aa3b2";
    p.push(`<text class="tok-label" x="${outLabelX}" y="${y + 13}" text-anchor="end" fill="${isT ? "#ff5c7a" : "#5c6675"}">${tok(id)}</text>`);
    p.push(`<rect x="${outBarX}" y="${y + 3}" width="${outBarW}" height="15" fill="#eef1f8" rx="4"/>`);
    p.push(`<rect x="${outBarX}" y="${y + 3}" width="${Math.max(1, tr.probs[id] * outBarW)}" height="15" fill="${c}" rx="4"/>`);
    p.push(`<text class="tok-label" x="${outBarX + outBarW + 6}" y="${y + 15}" fill="#5c6675">${pct(tr.probs[id])}</text>`);
  });

  $("#netflow").innerHTML = p.join("");
}

/* =========================================================================
   output bars + stats
   ========================================================================= */
function drawBars(alpha) {
  const theta = thetaAt(art, alpha);
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
  $("#s-forget").textContent = pct(forgetTargetProb(art, theta));
  $("#s-retain").textContent = pct(retainAccuracy(art, theta));
}

/* =========================================================================
   trade-off curve
   ========================================================================= */
const CW = 560, CH = 240, PAD = 34, AMAX = 1.25;
const cx = (a) => PAD + (a / AMAX) * (CW - 2 * PAD);
const cy = (v) => CH - PAD - v * (CH - 2 * PAD);
const gridP = art.alpha_grid.map((a) => forgetTargetProb(art, thetaAt(art, a)));
const gridR = art.alpha_grid.map((a) => retainAccuracy(art, thetaAt(art, a)));
const poly = (vals, color) =>
  `<polyline points="${art.alpha_grid.map((a, i) => `${cx(a)},${cy(vals[i])}`).join(" ")}"
     fill="none" stroke="${color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>`;
function drawCurve(alpha) {
  const axes = `
    <line x1="${PAD}" y1="${cy(0)}" x2="${CW - PAD}" y2="${cy(0)}" stroke="#e7e9f0"/>
    <line x1="${PAD}" y1="${cy(1)}" x2="${CW - PAD}" y2="${cy(1)}" stroke="#f1f2f7"/>
    <text x="${PAD - 6}" y="${cy(1) + 4}" text-anchor="end" font-size="10" fill="#9aa3b2" font-family="monospace">1.0</text>
    <text x="${PAD - 6}" y="${cy(0) + 4}" text-anchor="end" font-size="10" fill="#9aa3b2" font-family="monospace">0.0</text>
    <text x="${cx(0)}" y="${CH - 8}" text-anchor="middle" font-size="10" fill="#9aa3b2" font-family="monospace">α=0</text>
    <text x="${cx(AMAX)}" y="${CH - 8}" text-anchor="middle" font-size="10" fill="#9aa3b2" font-family="monospace">α=1.25</text>`;
  const th = thetaAt(art, alpha);
  const marker = `
    <line x1="${cx(alpha)}" y1="${PAD - 8}" x2="${cx(alpha)}" y2="${CH - PAD}" stroke="#c9cede" stroke-dasharray="3 3"/>
    <circle cx="${cx(alpha)}" cy="${cy(forgetTargetProb(art, th))}" r="5" fill="#ff5c7a" stroke="#fff" stroke-width="2"/>
    <circle cx="${cx(alpha)}" cy="${cy(retainAccuracy(art, th))}" r="5" fill="#1fb87a" stroke="#fff" stroke-width="2"/>`;
  $("#curve").innerHTML = axes + poly(gridP, "#ff5c7a") + poly(gridR, "#1fb87a") + marker;
}

/* ---------- master update ---------- */
function render(alpha) {
  drawVec(alpha);
  drawNet(alpha);
  drawBars(alpha);
  drawCurve(alpha);

  const th = thetaAt(art, alpha);
  const pf = forgetTargetProb(art, th), ra = retainAccuracy(art, th);
  $("#alpha-val").textContent = alpha.toFixed(2);
  $("#c-alpha").textContent = "α = " + alpha.toFixed(2);
  let msg;
  if (alpha < 0.05) msg = "The untouched model. Kyiv is the confident answer.";
  else if (pf < 0.05 && ra > 0.9) msg = "Sweet spot: Kyiv is gone, other capitals intact.";
  else if (pf < 0.05 && ra > 0.4) msg = "Kyiv suppressed, but collateral damage is starting.";
  else if (ra <= 0.4) msg = "Over-subtracted: retained knowledge is collapsing too.";
  else msg = "Kyiv's probability is draining into other capitals.";
  $("#c-msg").textContent = msg;

  $("#ticks").querySelectorAll("span").forEach((el) =>
    el.classList.toggle("hot", Math.abs(parseFloat(el.dataset.a) - alpha) < 0.06));
}
function setAlpha(a) { $("#alpha").value = a; render(a); }
$("#alpha").addEventListener("input", (e) => render(parseFloat(e.target.value)));

/* ---------- honesty self-test ---------- */
const diff = selfTest(art);
const st = $("#selftest");
if (diff < 1e-6) { st.classList.add("ok");
  st.textContent = `live inference matches the reference pipeline (max Δ = ${diff.toExponential(1)})`;
} else st.textContent = `⚠ live/reference mismatch: Δ = ${diff.toExponential(2)}`;

if ($("#archdiag")) drawArch($("#archdiag"), art, { mode: "all" });
render(0);
