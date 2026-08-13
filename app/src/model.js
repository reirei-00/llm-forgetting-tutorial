// Live TRANSFORMER inference in the browser.
// This forward pass mirrors pipeline/tformer.py exactly: token+pos embeddings,
// N pre-LayerNorm blocks (causal multi-head attention + GELU MLP), final LayerNorm,
// tied output projection. Nothing is faked; theta(alpha)=theta_full-alpha*v_f is
// computed element-wise, then the real transformer runs.

export function loadArtifact(url) { return fetch(url).then((r) => r.json()); }

// theta(alpha) = theta_full - alpha * v_f, per named parameter tensor.
export function thetaAt(art, alpha) {
  const full = art.theta_full, vf = art.v_f, out = {};
  for (const k of art.param_keys) {
    const a = full[k], b = vf[k], n = a.length, o = new Float64Array(n);
    for (let i = 0; i < n; i++) o[i] = a[i] - alpha * b[i];
    out[k] = o;
  }
  return out;
}

function linear(vec, W, b, din, dout) {           // out[k] = b[k] + sum_j vec[j]*W[j*dout+k]
  const o = new Float64Array(dout);
  for (let k = 0; k < dout; k++) { let s = b ? b[k] : 0; for (let j = 0; j < din; j++) s += vec[j] * W[j * dout + k]; o[k] = s; }
  return o;
}
function layernorm(v, g, b, d, eps) {
  let mu = 0; for (let j = 0; j < d; j++) mu += v[j]; mu /= d;
  let va = 0; for (let j = 0; j < d; j++) { const t = v[j] - mu; va += t * t; } va /= d;
  const inv = 1 / Math.sqrt(va + eps), o = new Float64Array(d);
  for (let j = 0; j < d; j++) o[j] = (v[j] - mu) * inv * g[j] + b[j];
  return o;
}
const gelu = (x) => 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));

// Full forward. Returns embeddings, per-block/head attention, final hidden, logits, probs.
export function forwardTrace(art, theta, ids) {
  const { d, heads, blocks, d_ff, d_head: dh, seq_len: T, vocab_size: Vn, eps } = art.meta;
  const P = (bi, nm) => theta[`b${bi}.${nm}`];
  let x = [], emb = [];
  for (let t = 0; t < T; t++) {
    const row = new Float64Array(d), tok = ids[t];
    for (let j = 0; j < d; j++) row[j] = theta.tok[tok * d + j] + theta.pos[t * d + j];
    x.push(row); emb.push(row.slice());
  }
  const attnAll = [];
  for (let bi = 0; bi < blocks; bi++) {
    const xln = x.map((r) => layernorm(r, P(bi, "ln1_g"), P(bi, "ln1_b"), d, eps));
    const q = xln.map((r) => linear(r, P(bi, "Wq"), P(bi, "bq"), d, d));
    const k = xln.map((r) => linear(r, P(bi, "Wk"), P(bi, "bk"), d, d));
    const v = xln.map((r) => linear(r, P(bi, "Wv"), P(bi, "bv"), d, d));
    const ctx = x.map(() => new Float64Array(d)), heads_a = [];
    for (let h = 0; h < heads; h++) {
      const off = h * dh, amat = [];
      for (let i = 0; i < T; i++) {
        const sc = new Float64Array(i + 1);
        for (let kk = 0; kk <= i; kk++) { let s = 0; for (let c = 0; c < dh; c++) s += q[i][off + c] * k[kk][off + c]; sc[kk] = s / Math.sqrt(dh); }
        let mx = -Infinity; for (let kk = 0; kk <= i; kk++) if (sc[kk] > mx) mx = sc[kk];
        let sum = 0; const e = new Float64Array(i + 1); for (let kk = 0; kk <= i; kk++) { e[kk] = Math.exp(sc[kk] - mx); sum += e[kk]; }
        const arow = new Array(T).fill(0);
        for (let kk = 0; kk <= i; kk++) { const a = e[kk] / sum; arow[kk] = a; for (let c = 0; c < dh; c++) ctx[i][off + c] += a * v[kk][off + c]; }
        amat.push(arow);
      }
      heads_a.push(amat);
    }
    attnAll.push(heads_a);
    for (let i = 0; i < T; i++) { const o = linear(ctx[i], P(bi, "Wo"), P(bi, "bo"), d, d); for (let j = 0; j < d; j++) x[i][j] += o[j]; }
    for (let i = 0; i < T; i++) {
      const h2 = layernorm(x[i], P(bi, "ln2_g"), P(bi, "ln2_b"), d, eps);
      const hid = linear(h2, P(bi, "fc1_W"), P(bi, "fc1_b"), d, d_ff);
      for (let z = 0; z < d_ff; z++) hid[z] = gelu(hid[z]);
      const out = linear(hid, P(bi, "fc2_W"), P(bi, "fc2_b"), d_ff, d);
      for (let j = 0; j < d; j++) x[i][j] += out[j];
    }
  }
  const hidden = x.map((r) => layernorm(r, theta.ln_f_g, theta.ln_f_b, d, eps));
  const last = hidden[T - 1], logits = new Float64Array(Vn);
  for (let w = 0; w < Vn; w++) { let s = 0; for (let j = 0; j < d; j++) s += last[j] * theta.tok[w * d + j]; logits[w] = s; }
  return { emb, attn: attnAll, hidden, logits, probs: softmax(logits) };
}

export function softmax(z) {
  let m = -Infinity; for (const x of z) if (x > m) m = x;
  const e = new Float64Array(z.length); let s = 0;
  for (let i = 0; i < z.length; i++) { e[i] = Math.exp(z[i] - m); s += e[i]; }
  for (let i = 0; i < e.length; i++) e[i] /= s; return e;
}

export function probsFor(art, theta, ids) { return forwardTrace(art, theta, ids).probs; }
export function forgetTargetProb(art, theta) { return probsFor(art, theta, art.prompts.forget.ids)[art.prompts.forget.answer_id]; }
export function retainAccuracy(art, theta) {
  let ok = 0;
  for (const r of art.prompts.retain) {
    const p = probsFor(art, theta, r.ids); let bi = 0, best = -1;
    for (let i = 0; i < p.length; i++) if (p[i] > best) { best = p[i]; bi = i; }
    if (bi === r.answer_id) ok++;
  }
  return ok / art.prompts.retain.length;
}
export function topk(art, theta, ids, k = 6) {
  const p = probsFor(art, theta, ids);
  const idx = Array.from(p.keys()).sort((a, b) => p[b] - p[a]);
  const top = idx.slice(0, k).map((i) => ({ token: art.vocab.tokens[i], id: i, p: p[i] }));
  return { top, other: Math.max(0, 1 - top.reduce((s, t) => s + t.p, 0)) };
}
export function projectAlpha(art, alpha) {
  const g = art.geometry;
  return [g.theta_full[0] - alpha * g.vf[0], g.theta_full[1] - alpha * g.vf[1]];
}
export function selfTest(art) {
  let maxDiff = 0;
  for (const ref of art.reference) {
    const th = thetaAt(art, ref.alpha);
    maxDiff = Math.max(maxDiff, Math.abs(forgetTargetProb(art, th) - ref.p_target));
    maxDiff = Math.max(maxDiff, Math.abs(retainAccuracy(art, th) - ref.retain_acc));
  }
  return maxDiff;
}
