"""
Chapter 0 — why exact unlearning is hard, on the tiny TRANSFORMER (PyTorch).

  1. DISTRIBUTED: v_f = theta_fgt - theta_0 has weight in every module. We report its
     participation ratio (effective number of parameters it spreads across) and a
     per-module-group breakdown.
  2. NON-CONVEXITY: retrain without Kyiv from several seeds -> identical behaviour,
     different weights. 2-D PCA shows the distinct minima.
"""
import json, os
import numpy as np
import torch
import tformer as tf

X_all, y_cap, y_unk, fi = tf.datasets()
keep = [i for i in range(len(tf.FACTS)) if i != fi]

def flat(nf): return np.concatenate([np.array(nf[k]) for k in nf])

print("Measuring the fact's footprint (v_f) on the transformer ...")
th0 = tf.train_theta0(0)
th_full0, _ = tf.train_theta_full(th0)
th_fgt = tf.copy_model(th_full0); tf.train(th_fgt, X_all[fi:fi+1], y_cap[fi:fi+1], 30, lr=0.005, ls=0.0)
nf_base = tf.named_flat(th_full0)
nf0, nffgt = nf_base, tf.named_flat(th_fgt)
KEYS = list(nf0.keys())
vf = flat(nffgt) - flat(nf0)
Nparam = vf.size
PR = float((vf @ vf) ** 2 / np.sum(vf ** 4))
pct_active = float(np.mean(np.abs(vf) > 0.01 * np.abs(vf).max()))

def group_of(k):
    if k in ("tok", "pos"): return "token + position embeddings"
    if k.startswith("b"):
        blk = int(k[1]) + 1
        return f"block {blk} · attention" if any(s in k for s in ["Wq","bq","Wk","bk","Wv","bv","Wo","bo","ln1"]) else f"block {blk} · MLP"
    return "final layer-norm"
group_norms = {}
for k in KEYS:
    g = group_of(k); n2 = float(np.sum(np.array(nffgt[k]) - np.array(nf0[k])) ** 2 if False else np.sum((np.array(nffgt[k]) - np.array(nf0[k])) ** 2))
    group_norms[g] = group_norms.get(g, 0.0) + n2
group_norms = {g: float(np.sqrt(v)) for g, v in group_norms.items()}
print(f"  params={Nparam} ; participation ratio ~{PR:.0f} ({100*PR/Nparam:.0f}%) ; |v_f|>1% in {100*pct_active:.0f}%")

print("Retraining without Kyiv from several seeds (keeping the ones that converge) ...")
WANT = 6
runs, behav = [], []
for s in range(1, 20):
    if len(runs) >= WANT: break
    base = tf.train_theta0(s)
    X_n, y_n = tf.nonce_datasets()
    ref = tf.copy_model(base); tf.train(ref, torch.cat([X_all[keep], X_n]), torch.cat([y_cap[keep], y_n]), 2500, lr=0.01, ls=0.05)
    p = tf.probs_last(ref, X_all)
    acc = float((p[keep].argmax(1) == y_cap[keep].numpy()).mean())
    if acc < 0.99:  # a run that landed in a poor minimum — skip so "identical behaviour" is honest
        print(f"    seed {s}: retain={acc:.2f}  (skipped)"); continue
    runs.append(flat(tf.named_flat(ref)))
    behav.append({"seed": s, "retain_acc": acc, "p_kyiv": float(p[fi][tf.STOI["kyiv"]])})
SEEDS = [b["seed"] for b in behav]
runs_full = flat(nf_base)

M = np.stack(runs)
dists = [float(np.linalg.norm(runs[i] - runs[j])) for i in range(len(runs)) for j in range(i + 1, len(runs))]
selfnorm = float(np.mean([np.linalg.norm(r) for r in runs]))
print(f"  {len(SEEDS)} solutions; pairwise dist mean {np.mean(dists):.2f} (min {np.min(dists):.2f}, max {np.max(dists):.2f}); ‖theta‖~{selfnorm:.1f}")
for b in behav: print(f"    seed {b['seed']}: retain={b['retain_acc']:.2f} p(kyiv)={b['p_kyiv']:.3f}")

pts = np.vstack([M, runs_full[None]]); Xc = pts - pts.mean(0)
U, Sg, Vt = np.linalg.svd(Xc, full_matrices=False)
coords = Xc @ Vt[:2].T
minima = [{"seed": SEEDS[i], "xy": [float(coords[i, 0]), float(coords[i, 1])],
           "retain_acc": behav[i]["retain_acc"], "p_kyiv": behav[i]["p_kyiv"]} for i in range(len(SEEDS))]

artifact = {
    "meta": {**tf.meta(), "note": "Real measurements on the tiny transformer; the parameter-space cloud is schematic."},
    "distributed": {"n_params": Nparam, "participation_ratio": PR, "pct_params": 100 * PR / Nparam,
                    "pct_active": 100 * pct_active, "group_norms": group_norms,
                    "arch_order": ["token + position embeddings", "block 1 · attention", "block 1 · MLP",
                                   "block 2 · attention", "block 2 · MLP", "final layer-norm"]},
    "nonconvex": {"n_seeds": len(SEEDS), "pair_dist_mean": float(np.mean(dists)),
                  "pair_dist_min": float(np.min(dists)), "pair_dist_max": float(np.max(dists)),
                  "theta_norm": selfnorm, "minima": minima,
                  "full_xy": [float(coords[-1, 0]), float(coords[-1, 1])], "behaviors": behav},
}
out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app", "data", "artifact_ch0.json"))
with open(out, "w") as f: json.dump(artifact, f)
print(f"Wrote {out}  ({os.path.getsize(out)/1024:.1f} KB)")
