// Transformer architecture schematic — the canonical vertical form.
//
// Same layout everyone knows from "Attention Is All You Need" (Fig. 1), reduced to
// the decoder-only stack this tutorial actually trains:
//
//   tokens → embedding → ⊕ positional → [ Masked MHA → Add & Norm →
//   Feed Forward → Add & Norm ] × 2 → LayerNorm → Linear (tied) → Softmax
//
// Residual connections are drawn as skip arrows around each sub-layer into its
// "Add & Norm" — which is what makes Chapter 3 legible: activation steering adds a
// vector onto that same residual path, alongside what the sub-layers already add.
//
// mode: "none" | "footprint" | "all" (Ch1) | "grads" (Ch2) | "steer" | "baked" (Ch3)

export function drawArch(svg, art, opts = {}) {
  const mode = opts.mode || "none";
  const layer = opts.layer ?? 1;
  const m = art.meta || {};
  const d = m.d ?? 32, heads = m.heads ?? 2, blocks = m.blocks ?? 2;
  const dff = m.d_ff ?? 64, V = m.vocab_size ?? 30, T = m.seq_len ?? 5;
  const bakedLayer = art.baked_info?.layer;
  const fp = opts.footprint || null;
  const fpMax = fp ? Math.max(...Object.values(fp)) : 1;

  const VIO = "#7c5cff", CORAL = "#ff5c7a", INK = "#1c2230", MUT = "#6b7486", LINE = "#d7dded";
  const W = 560, CX = 300, BW = 232;
  const P = [];
  let y = 34;

  const box = (h, title, sub, o = {}) => {
    const fill = o.fill || "#f4f6fb", stroke = o.stroke || "#dde3ee";
    P.push(`<rect x="${CX - BW / 2}" y="${y}" width="${BW}" height="${h}" rx="8" fill="${fill}" stroke="${stroke}" stroke-width="${o.bold ? 1.9 : 1.2}"/>`);
    P.push(`<text x="${CX}" y="${y + (sub ? h / 2 - 3 : h / 2 + 4)}" text-anchor="middle" font-size="12" font-weight="600" fill="${o.ink || INK}" font-family="monospace">${title}</text>`);
    if (sub) P.push(`<text x="${CX}" y="${y + h / 2 + 12}" text-anchor="middle" font-size="9.5" fill="${MUT}" font-family="monospace">${sub}</text>`);
    if (o.note) P.push(`<text x="${CX + BW / 2 + 10}" y="${y + h / 2 + 4}" font-size="10" fill="${o.noteColor || VIO}" font-family="monospace">${o.note}</text>`);
    const top = y; y += h; return top;
  };
  const arrow = (len = 16) => {
    P.push(`<line x1="${CX}" y1="${y}" x2="${CX}" y2="${y + len - 6}" stroke="${LINE}" stroke-width="1.6"/>`);
    P.push(`<path d="M${CX},${y + len} l-4.5,-7 l9,0 z" fill="${LINE}"/>`);
    y += len;
  };
  const skip = (fromY, toY, color = LINE) => {
    const rx = CX - BW / 2 - 26;
    P.push(`<path d="M${CX - BW / 2},${fromY} H${rx} V${toY} H${CX - BW / 2 - 2}" fill="none" stroke="${color}" stroke-width="1.5" stroke-dasharray="4 3"/>`);
    P.push(`<path d="M${CX - BW / 2 - 1},${toY} l-7,-4.5 l0,9 z" fill="${color}"/>`);
  };

  // ---------------- output head (top) ----------------
  box(38, "next-word probabilities", `over ${V} words`);
  arrow();
  box(32, "Softmax");
  arrow();
  box(34, "Linear (tied embeddings)", `${d} → ${V}`);
  arrow();
  box(32, "Layer Norm");
  arrow();

  // ---------------- transformer blocks (top of page = last block) ----------------
  const steerMarks = [];
  for (let bi = blocks - 1; bi >= 0; bi--) {
    if (mode === "steer" && bi === layer) { y += 26; steerMarks.push(y - 15); }
    const gTop = y - 10;

    const ffState = (mode === "baked" && bi === bakedLayer) ? "baked"
      : (mode === "all" || mode === "grads") ? "on" : "off";
    const atState = (mode === "all" || mode === "grads") ? "on" : "off";
    const tint = (kind) => {
      if (mode === "footprint" && fp) {
        const key = `block ${bi + 1} · ${kind === "attn" ? "attention" : "MLP"}`;
        const v = (fp[key] ?? 0) / fpMax;
        return { fill: `rgba(124,92,255,${(0.08 + 0.32 * v).toFixed(3)})`, stroke: "#b9a8f7", bold: true,
                 note: `‖v_f‖ ${(fp[key] ?? 0).toFixed(2)}` };
      }
      const st = kind === "attn" ? atState : ffState;
      if (st === "baked") return { fill: "#ffe6ec", stroke: CORAL, bold: true, note: "re-solved", noteColor: CORAL };
      if (st === "on") return { fill: "#f1edff", stroke: "#b9a8f7", bold: true };
      return {};
    };

    box(30, "Add & Norm");
    const anFF = y - 30;
    arrow(14);
    box(40, "Feed Forward", `${d} → ${dff} → ${d} · GELU`, tint("mlp"));
    arrow(14);
    box(30, "Add & Norm");
    const anAT = y - 30;
    arrow(14);
    const atTop = box(44, "Masked Multi-Head Attention", `${heads} heads · d_head ${d / heads}`, tint("attn"));

    skip(atTop + 50, anFF + 15, ffState !== "off" ? "#c9bdf5" : LINE);
    skip(atTop + 50, anAT + 15, LINE);

    const gBot = y + 6;
    P.push(`<path d="M${CX + BW / 2 + 16},${gTop} h10 V${gBot} h-10" fill="none" stroke="#e3e8f2" stroke-width="1.3"/>`);
    P.push(`<text x="${CX + BW / 2 + 32}" y="${(gTop + gBot) / 2}" font-size="11" font-weight="600" fill="${MUT}" font-family="monospace">block ${bi + 1}</text>`);
    if (bi > 0) arrow();
  }

  // ---------------- input (bottom) ----------------
  arrow();
  const posY = y + 13;
  P.push(`<circle cx="${CX}" cy="${posY}" r="13" fill="#fff" stroke="${LINE}" stroke-width="1.6"/>`);
  P.push(`<line x1="${CX - 6}" y1="${posY}" x2="${CX + 6}" y2="${posY}" stroke="${MUT}" stroke-width="1.6"/>`);
  P.push(`<line x1="${CX}" y1="${posY - 6}" x2="${CX}" y2="${posY + 6}" stroke="${MUT}" stroke-width="1.6"/>`);
  P.push(`<text x="${CX + 22}" y="${posY + 4}" font-size="10.5" fill="${MUT}" font-family="monospace">positional encoding</text>`);
  y = posY + 13; arrow();
  box(34, "Token Embedding", `${V} × ${d}`);
  arrow();
  box(32, `input tokens · ${T}`, `"the capital of ukraine is"`);

  // ---------------- steering marker on the residual path ----------------
  for (const sy of steerMarks) {
    P.push(`<circle cx="${CX}" cy="${sy}" r="13" fill="#fff" stroke="${CORAL}" stroke-width="2.4"/>`);
    P.push(`<line x1="${CX - 6}" y1="${sy}" x2="${CX + 6}" y2="${sy}" stroke="${CORAL}" stroke-width="2"/>`);
    P.push(`<line x1="${CX}" y1="${sy - 6}" x2="${CX}" y2="${sy + 6}" stroke="${CORAL}" stroke-width="2"/>`);
    P.push(`<line x1="${CX - 96}" y1="${sy}" x2="${CX - 15}" y2="${sy}" stroke="${CORAL}" stroke-width="1.6"/>`);
    P.push(`<path d="M${CX - 14},${sy} l-7,-4.5 l0,9 z" fill="${CORAL}"/>`);
    P.push(`<text x="${CX - 100}" y="${sy - 4}" text-anchor="end" font-size="11.5" fill="${CORAL}" font-family="monospace">+ β·r</text>`);
    P.push(`<text x="${CX - 100}" y="${sy + 10}" text-anchor="end" font-size="9.5" fill="${MUT}" font-family="monospace">steering vector</text>`);
  }

  // ---------------- caption ----------------
  const caps = {
    none:      ["the model this tutorial trains and runs", `decoder-only transformer · ${blocks} blocks · d=${d} · ${heads} heads · ~18.4k parameters`],
    footprint: ["the Kyiv fact is spread across the whole stack", "‖v_f‖ per sub-layer — near-identical everywhere: no layer owns the fact"],
    all:       ["task-vector negation edits every violet sub-layer", "one subtraction θ_full − α·v_f, applied across the whole model"],
    grads:     ["gradient ascent updates every violet sub-layer", "the ascent gradient flows back through attention and feed-forward alike"],
    steer:     ["no weights change at all", "a vector is added onto the residual path — where the sub-layers already write"],
    baked:     ["one coral matrix is re-solved", "the redirection is written into a single feed-forward down-projection"],
  }[mode] || ["", ""];
  const H = y + 46;
  P.unshift(`<text x="${CX}" y="18" text-anchor="middle" font-size="12.5" font-weight="600" fill="${INK}">${caps[0]}</text>`);
  P.push(`<text x="${CX}" y="${H - 14}" text-anchor="middle" font-size="10.5" fill="${MUT}">${caps[1]}</text>`);

  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.innerHTML = P.join("");
}
