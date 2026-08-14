// Shared transformer architecture diagram.
// Draws the actual model — tokens → embeddings → [attention → MLP] × 2 → LayerNorm → output —
// with the residual stream as a spine, and highlights WHERE the current chapter intervenes.
//
// mode: "all"      (ch1: weight arithmetic touches every weight block)
//       "grads"    (ch2: optimization pushes gradients through every weight block)
//       "steer"    (ch3: a vector is injected into the residual stream after one block)
//       "baked"    (ch3: one MLP down-projection is re-solved)

export function drawArch(svg, art, opts = {}) {
  const mode = opts.mode || "all";
  const layer = opts.layer ?? 1;
  const { d, heads, blocks, d_ff, vocab_size: V, seq_len: T } = art.meta;

  const W = 900, H = 250, Y = 96, BH = 46;          // block row geometry
  const boxW = 118, gap = 26, x0 = 92;
  const P = [];
  const HL = { attn: "#7c5cff", mlp: "#7c5cff", none: "#e2e6f0" };

  const isTouched = (kind) => mode === "all" || mode === "grads";
  const fill = (kind, bi) => {
    if (mode === "baked" && kind === "mlp" && bi === art.baked_info?.layer) return "#ffe3ea";
    if (isTouched(kind)) return "#f1edff";
    return "#f7f8fc";
  };
  const stroke = (kind, bi) => {
    if (mode === "baked" && kind === "mlp" && bi === art.baked_info?.layer) return "#ff5c7a";
    if (isTouched(kind)) return "#c9bdf5";
    return "#e2e6f0";
  };

  // residual stream spine
  P.push(`<line x1="${x0 - 40}" y1="${Y + BH / 2}" x2="${W - 150}" y2="${Y + BH / 2}"
     stroke="#dfe3ec" stroke-width="10" stroke-linecap="round"/>`);
  P.push(`<text x="${x0 - 78}" y="${Y - 16}" font-size="10.5" fill="#9aa3b2" font-family="monospace">residual stream (${d}-dim) — carried through every block</text>`);

  // input tokens
  P.push(box(x0 - 78, Y, 62, BH, "#eef1f8", "#e2e6f0", "tokens", `${T} words`));

  let x = x0;
  for (let bi = 0; bi < blocks; bi++) {
    P.push(box(x, Y, boxW, BH, fill("attn", bi), stroke("attn", bi),
      `attention ${bi + 1}`, `${heads} heads · causal`));
    x += boxW + gap;
    P.push(box(x, Y, boxW, BH, fill("mlp", bi), stroke("mlp", bi),
      `MLP ${bi + 1}`, `${d}→${d_ff}→${d}`));
    // steering injection marker sits AFTER the block (i.e. after this MLP)
    if (mode === "steer" && bi === layer) {
      const ix = x + boxW + gap / 2;
      P.push(`<circle cx="${ix}" cy="${Y + BH / 2}" r="13" fill="#fff" stroke="#ff5c7a" stroke-width="2.5"/>`);
      P.push(`<text x="${ix}" y="${Y + BH / 2 + 4}" text-anchor="middle" font-size="12" fill="#ff5c7a" font-family="monospace">+</text>`);
      P.push(`<line x1="${ix}" y1="${Y + BH + 34}" x2="${ix}" y2="${Y + BH / 2 + 15}" stroke="#ff5c7a" stroke-width="1.6"/>`);
      P.push(`<path d="M${ix},${Y + BH / 2 + 14} l-4,7 l8,0 z" fill="#ff5c7a"/>`);
      P.push(`<text x="${ix}" y="${Y + BH + 50}" text-anchor="middle" font-size="11.5" fill="#ff5c7a" font-family="monospace">h + β·r</text>`);
      P.push(`<text x="${ix}" y="${Y + BH + 64}" text-anchor="middle" font-size="9.5" fill="#9aa3b2" font-family="monospace">injected here</text>`);
    }
    x += boxW + gap;
  }
  P.push(box(x, Y, 74, BH, "#eef1f8", "#e2e6f0", "LayerNorm", "+ tied out"));
  x += 74 + gap;
  P.push(box(x, Y, 84, BH, "#eef1f8", "#e2e6f0", "softmax", `${V} words`));

  // caption
  const caps = {
    all:   ["every violet block is edited", "task-vector negation subtracts v_f from all weights at once"],
    grads: ["every violet block receives gradient", "ascent updates all weights; the anchor limits the damage"],
    steer: ["no weight changes at all", "one vector is added to the residual stream at inference time"],
    baked: ["one coral matrix is re-solved", "the redirection is written into a single MLP down-projection"],
    none:  ["the model every chapter intervenes on", "2 causal attention+MLP blocks, joined by the residual stream"],
  }[mode] || ["the model every chapter intervenes on", ""];
  P.push(`<text x="${W / 2}" y="30" text-anchor="middle" font-size="12" font-weight="600" fill="#1c2230">${caps[0]}</text>`);
  P.push(`<text x="${W / 2}" y="48" text-anchor="middle" font-size="11" fill="#5c6675">${caps[1]}</text>`);
  P.push(`<text x="${W / 2}" y="${H - 12}" text-anchor="middle" font-size="10" fill="#c2c8d4" font-family="monospace">the real model: ${blocks} blocks · d=${d} · ${heads} heads · ~18.4k parameters</text>`);

  svg.innerHTML = P.join("");

  function box(bx, by, bw, bh, f, st, title, sub) {
    return `<rect x="${bx}" y="${by}" width="${bw}" height="${bh}" rx="9" fill="${f}" stroke="${st}" stroke-width="1.5"/>
      <text x="${bx + bw / 2}" y="${by + 20}" text-anchor="middle" font-size="11.5" font-weight="600" fill="#1c2230" font-family="monospace">${title}</text>
      <text x="${bx + bw / 2}" y="${by + 35}" text-anchor="middle" font-size="9.5" fill="#5c6675" font-family="monospace">${sub}</text>`;
  }
}
