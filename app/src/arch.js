// Shared transformer schematic.
//
// Visual grammar follows the standard used by Transformer Explainer (poloclub),
// bbycroft.net/llm and the mechanistic-interpretability "circuits" convention:
// the RESIDUAL STREAM is a continuous highway, and each sub-layer *branches off*
// it, computes, and *adds back* through a ⊕ node. Nothing is drawn inline.
//
// That grammar earns its keep here: activation steering is drawn as one more ⊕ on
// the same highway — visually identical to what attention and MLP already do,
// which is exactly the point of Chapter 3.
//
// mode: "none"  — just the architecture (Ch0)
//       "all"   — every weight block is edited (Ch1, task-vector negation)
//       "grads" — every weight block receives gradient (Ch2, gradient ascent)
//       "steer" — a vector is added to the stream after one block (Ch3)
//       "baked" — one MLP down-projection is re-solved (Ch3, LUNAR)

export function drawArch(svg, art, opts = {}) {
  const mode = opts.mode || "none";
  const layer = opts.layer ?? 1;
  const m = art.meta || {};
  const d = m.d ?? 32, heads = m.heads ?? 2, blocks = m.blocks ?? 2;
  const dff = m.d_ff ?? 64, V = m.vocab_size ?? 30, T = m.seq_len ?? 5;
  const bakedLayer = art.baked_info?.layer;

  const W = 900, H = 322;
  const SY = 226;                 // residual stream y
  const BY = 92, BH = 62;         // sub-layer box band
  const P = [];

  const CORAL = "#ff5c7a", GREY = "#c8cfdc";
  const fp = opts.footprint || null;                 // {"block 1 · attention": norm, ...}
  const fpMax = fp ? Math.max(...Object.values(fp)) : 1;
  const fpKey = (kind, bi) => `block ${bi + 1} · ${kind === "attn" ? "attention" : "MLP"}`;
  const fpTint = (kind, bi) => {
    const v = (fp?.[fpKey(kind, bi)] ?? 0) / fpMax;
    return [`rgba(124,92,255,${(0.06 + 0.34 * v).toFixed(3)})`, "#b9a8f7"];
  };
  const edited = (kind, bi) =>
    (mode === "all" || mode === "grads") ? "on"
      : (mode === "baked" && kind === "mlp" && bi === bakedLayer) ? "baked" : "off";
  const paint = { on: ["#f1edff", "#b9a8f7"], baked: ["#ffe6ec", CORAL], off: ["#fbfcff", "#e4e8f1"] };

  // ---------- flow layout: place elements along the stream with a cursor ----------
  const SLOT = 116, GAP = 14, STEER_W = 40;
  const items = [];
  let x = 96;
  for (let bi = 0; bi < blocks; bi++) {
    for (const kind of ["attn", "mlp"]) {
      items.push({ type: "sub", kind, bi, x0: x, x1: x + SLOT });
      x += SLOT + GAP;
      if (mode === "steer" && kind === "mlp" && bi === layer) {
        items.push({ type: "steer", cx: x + STEER_W / 2 - GAP / 2 });
        x += STEER_W;
      }
    }
  }
  const lnX = x + 6, lnW = 78;
  const outX = lnX + lnW + GAP + 6, outW = 92;

  // ---------- the residual stream (drawn first; everything sits on it) ----------
  P.push(`<line x1="84" y1="${SY}" x2="${outX - 6}" y2="${SY}" stroke="#dde3ee" stroke-width="11" stroke-linecap="round"/>`);
  P.push(`<line x1="84" y1="${SY}" x2="${outX - 6}" y2="${SY}" stroke="#eef2f9" stroke-width="5" stroke-linecap="round"/>`);
  P.push(`<text x="88" y="${SY + 30}" font-size="10.5" fill="#8a93a5" font-family="monospace">residual stream · ${d}-dim · every layer reads from it and adds back into it</text>`);

  // ---------- block grouping brackets (the repeating pattern) ----------
  for (let bi = 0; bi < blocks; bi++) {
    const subs = items.filter((i) => i.type === "sub" && i.bi === bi);
    const a = subs[0].x0 - 6, b = subs[subs.length - 1].x1 + 6;
    P.push(`<path d="M${a},${BY - 16} L${a},${BY - 24} L${b},${BY - 24} L${b},${BY - 16}" fill="none" stroke="#dfe4ee" stroke-width="1.2"/>`);
    P.push(`<text x="${(a + b) / 2}" y="${BY - 29}" text-anchor="middle" font-size="10.5" fill="#8a93a5" font-family="monospace">transformer block ${bi + 1}</text>`);
  }

  // ---------- input / output chips ----------
  P.push(chip(14, SY - 19, 70, 38, "#eef1f8", "#e0e5ef", "tokens", `${T} words`));
  P.push(chip(lnX, SY - 19, lnW, 38, "#eef1f8", "#e0e5ef", "LayerNorm", "+ unembed"));
  P.push(chip(outX, SY - 19, outW, 38, "#eef1f8", "#e0e5ef", "next word", `${V} probs`));

  // ---------- sub-layers: read → compute → ⊕ back ----------
  for (const it of items) {
    if (it.type !== "sub") continue;
    const rx = it.x0 + 14, wx = it.x1 - 14;
    const st = edited(it.kind, it.bi);
    const [fill, stroke] = mode === "footprint" ? fpTint(it.kind, it.bi) : paint[st];
    P.push(`<path d="M${rx},${SY - 6} C${rx},${SY - 46} ${it.x0 + 16},${BY + BH + 20} ${it.x0 + 22},${BY + BH}" fill="none" stroke="${st === "off" && mode !== "footprint" ? GREY : stroke}" stroke-width="1.6"/>`);
    P.push(`<path d="M${it.x1 - 22},${BY + BH} C${wx - 16},${BY + BH + 20} ${wx},${SY - 46} ${wx},${SY - 12}" fill="none" stroke="${st === "off" && mode !== "footprint" ? GREY : stroke}" stroke-width="1.6"/>`);
    const title = it.kind === "attn" ? `attention ${it.bi + 1}` : `MLP ${it.bi + 1}`;
    const sub = it.kind === "attn" ? `${heads} heads · causal` : `${d}→${dff}→${d}`;
    P.push(chip(it.x0, BY, it.x1 - it.x0, BH, fill, stroke, title, sub, st !== "off" || mode === "footprint"));
    if (mode === "footprint" && fp)
      P.push(`<text x="${(it.x0 + it.x1) / 2}" y="${BY + BH + 15}" text-anchor="middle" font-size="9.5" fill="#7c5cff" font-family="monospace">‖v_f‖ ${(fp[fpKey(it.kind, it.bi)] ?? 0).toFixed(2)}</text>`);
    if (st === "baked")
      P.push(`<text x="${(it.x0 + it.x1) / 2}" y="${BY + BH + 15}" text-anchor="middle" font-size="9.5" fill="${CORAL}" font-family="monospace">down-proj re-solved</text>`);
    P.push(plus(wx, SY, st === "off" && mode !== "footprint" ? "#aab3c4" : stroke));
  }

  // ---------- the steering injection: one more ⊕ on the same highway ----------
  for (const it of items) {
    if (it.type !== "steer") continue;
    P.push(plus(it.cx, SY, CORAL, true));
    P.push(`<line x1="${it.cx}" y1="${SY + 44}" x2="${it.cx}" y2="${SY + 14}" stroke="${CORAL}" stroke-width="1.6"/>`);
    P.push(`<path d="M${it.cx},${SY + 13} l-4,7 l8,0 z" fill="${CORAL}"/>`);
    P.push(`<text x="${it.cx}" y="${SY + 60}" text-anchor="middle" font-size="11.5" fill="${CORAL}" font-family="monospace">+ β·r</text>`);
    P.push(`<text x="${it.cx}" y="${SY + 74}" text-anchor="middle" font-size="9.5" fill="#8a93a5" font-family="monospace">steering vector</text>`);
  }

  // ---------- caption ----------
  const caps = {
    footprint: ["the Kyiv fact is spread across the whole model", "‖v_f‖ per block — near-identical everywhere: no block owns the fact"],
    none:  ["the model every chapter intervenes on", `${blocks} blocks of attention + MLP, joined by the residual stream`],
    all:   ["task-vector negation edits every violet block at once", "one subtraction θ_full − α·v_f applied across the whole model"],
    grads: ["gradient ascent updates every violet block", "the ascent gradient flows back through attention and MLP alike"],
    steer: ["no weights change at all", "a vector is added to the residual stream — exactly what the blocks themselves do"],
    baked: ["one coral matrix is re-solved", "the redirection is written into a single MLP down-projection"],
  }[mode] || ["the model every chapter intervenes on", ""];
  P.push(`<text x="${W / 2}" y="24" text-anchor="middle" font-size="12.5" font-weight="600" fill="#1c2230">${caps[0]}</text>`);
  P.push(`<text x="${W / 2}" y="42" text-anchor="middle" font-size="11" fill="#5c6675">${caps[1]}</text>`);
  P.push(`<text x="14" y="${H - 8}" text-anchor="start" font-size="9.5" fill="#c2c8d4" font-family="monospace">real model · ${blocks} blocks · d=${d} · ${heads} heads · ~18.4k params</text>`);

  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.innerHTML = P.join("");

  function chip(bx, by, bw, bh, f, st, title, sub, bold) {
    return `<rect x="${bx}" y="${by}" width="${bw}" height="${bh}" rx="9" fill="${f}" stroke="${st}" stroke-width="${bold ? 1.8 : 1.2}"/>
      <text x="${bx + bw / 2}" y="${by + bh / 2 - 2}" text-anchor="middle" font-size="11.5" font-weight="600" fill="#1c2230" font-family="monospace">${title}</text>
      <text x="${bx + bw / 2}" y="${by + bh / 2 + 13}" text-anchor="middle" font-size="9.5" fill="#6b7486" font-family="monospace">${sub}</text>`;
  }
  function plus(px, py, color, big) {
    const r = big ? 12 : 9.5;
    return `<circle cx="${px}" cy="${py}" r="${r}" fill="#fff" stroke="${color}" stroke-width="${big ? 2.4 : 1.8}"/>
      <line x1="${px - r / 2}" y1="${py}" x2="${px + r / 2}" y2="${py}" stroke="${color}" stroke-width="1.8"/>
      <line x1="${px}" y1="${py - r / 2}" x2="${px}" y2="${py + r / 2}" stroke="${color}" stroke-width="1.8"/>`;
  }
}
