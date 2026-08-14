"""
Chapter 3 — representation steering (class 3), on the tiny transformer.

Instead of rewriting weights via a loss, we move the model's INTERNAL STATE:

  r^(l) = mean residual activation on nonce ("unknown") prompts
        - residual activation on the forget prompt        (at block l, last position)

  inference-time steering:  h_last <- h_last + beta * r^(l)   for EVERY query
  LUNAR-style bake:         re-solve one MLP down-projection so forget activations
                            are redirected in the weights themselves (ridge, closed form)

Exports: steering vectors per layer, PCA basis for the activation view, baked weights,
and a reference metric sweep so the browser can self-test its live steered inference.
"""
import json, os
import numpy as np
import torch
import tformer as tf

torch.manual_seed(0)
X_all, y_cap, y_unk, fi = tf.datasets()
X_n, y_n = tf.nonce_datasets()
keep = [i for i in range(len(tf.FACTS)) if i != fi]
X_f = X_all[fi:fi+1]
UNK = tf.STOI["unknown"]; KYIV = tf.STOI["kyiv"]

print("Training theta_0 -> theta_full (with intact 'unknown' region) ...")
th0 = tf.train_theta0(0)
th_full, loss = tf.train_theta_full(th0)
p = tf.probs_last(th_full, torch.cat([X_all, X_n]))
acc12 = float((p[:12].argmax(1) == y_cap.numpy()).mean())
punk = float(np.mean([p[12+i][UNK] for i in range(len(tf.NONCE))]))
print(f"  full: loss={loss:.3f} capitals acc={acc12:.2f} p(unknown|nonce)={punk:.2f} p(kyiv|ukraine)={p[fi][KYIV]:.3f}")

# ---- steering vectors per layer ----
resid_all = tf.forward_residuals(th_full, torch.cat([X_all, X_n]))
LAYERS = list(range(tf.BLOCKS))
steer = {}
for l in LAYERS:
    h = resid_all[l][:, -1, :]
    steer[l] = (h[12:].mean(0) - h[fi])     # Ukraine-state -> unknown-region

def eval_steer(l, beta):
    pf = torch.softmax(tf.forward_steered(th_full, X_f, l, beta, steer[l])[:, -1, :], -1)[0]
    pr = torch.softmax(tf.forward_steered(th_full, X_all[keep], l, beta, steer[l])[:, -1, :], -1)
    acc = float((pr.argmax(1) == y_cap[keep]).float().mean())
    return float(pf[KYIV]), float(pf[UNK]), acc

BETAS = [round(b, 2) for b in np.arange(0.0, 2.001, 0.25).tolist()]
print("\nSteering sweep (p_kyiv / p_unknown / retain):")
ref_sweep = {}
for l in LAYERS:
    ref_sweep[l] = [dict(zip(("beta","p_kyiv","p_unknown","retain_acc"), (b,)+eval_steer(l, b))) for b in BETAS]
    row = " ".join(f"b={d['beta']}: {d['p_kyiv']:.2f}/{d['p_unknown']:.2f}/{d['retain_acc']:.2f}" for d in ref_sweep[l][::2])
    print(f"  layer {l}: {row}")

def score(l):
    d = next(x for x in ref_sweep[l] if x["beta"] == 1.0)
    return (1 - d["p_kyiv"]) + d["retain_acc"] + d["p_unknown"]
BEST = max(LAYERS, key=score)
print(f"\nSelected layer: {BEST}")

# ---- LUNAR-style bake at the selected layer ----
def fc2_inputs(model, idx, layer):
    with torch.no_grad():
        x = model.tok[idx] + model.pos[:idx.shape[1]]
        for i, blk in enumerate(model.blocks):
            ao, _ = blk.attn(tf.layernorm(x, blk.ln1_g, blk.ln1_b)); x = x + ao
            h = tf.gelu(tf.layernorm(x, blk.ln2_g, blk.ln2_b) @ blk.fc1_W + blk.fc1_b)
            if i == layer: return h[:, -1, :]
            x = x + (h @ blk.fc2_W + blk.fc2_b)

H_f = fc2_inputs(th_full, X_f, BEST)
H_keep = fc2_inputs(th_full, torch.cat([X_all[keep], X_n]), BEST)
th_baked = tf.bake_redirect(th_full, BEST, H_f, steer[BEST][None, :], H_keep, lam=1e-2)

with torch.no_grad():
    pb = torch.softmax(th_baked(X_f)[:, -1, :], -1)[0]
    pk_b, pu_b = float(pb[KYIV]), float(pb[UNK])
    pr_b = torch.softmax(th_baked(X_all[keep])[:, -1, :], -1)
    ra_b = float((pr_b.argmax(1) == y_cap[keep]).float().mean())
    delta_norm = float((th_baked.blocks[BEST].fc2_W - th_full.blocks[BEST].fc2_W).norm())
total = sum(v.numel() for v in th_full.parameters())
changed = th_full.blocks[BEST].fc2_W.numel()
print(f"baked: p(kyiv)={pk_b:.3f} p(unknown)={pu_b:.3f} retain={ra_b:.2f} | changed {changed}/{total} params, ||dW||={delta_norm:.2f}")

# ---- PCA basis (fit once on all last-position activations at BEST layer) ----
h = resid_all[BEST][:, -1, :].numpy()
mean = h.mean(0); Xc = h - mean
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
basis = Vt[:2]
coords = (Xc @ basis.T)
pca = {"mean": mean.tolist(), "basis": basis.tolist(),
       "points": {"forget": coords[fi].tolist(),
                  "retain": [coords[i].tolist() for i in keep],
                  "nonce": [coords[12 + i].tolist() for i in range(len(tf.NONCE))]},
       "labels": {"retain": [c for c, _ in tf.FACTS if c != tf.FORGET_COUNTRY], "nonce": tf.NONCE}}

artifact = {
    "meta": {**tf.meta(), "layers": LAYERS, "best_layer": BEST,
             "note": "Inference-time steering + LUNAR-style baked down-projection on the tiny transformer."},
    "vocab": {"tokens": tf.TOKENS, "stoi": tf.STOI},
    "param_keys": list(tf.named_flat(th_full).keys()),
    "theta_full": tf.named_flat(th_full),
    "theta_baked": tf.named_flat(th_baked),
    "baked_info": {"layer": BEST, "changed_params": changed, "total_params": total, "dW_norm": delta_norm,
                   "baked_metrics": {"p_kyiv": pk_b, "p_unknown": pu_b, "retain_acc": ra_b}},
    "steering": {str(l): steer[l].numpy().tolist() for l in LAYERS},
    "betas": BETAS,
    "reference": {str(l): ref_sweep[l] for l in LAYERS},
    "pca": pca,
    "prompts": {"template_tokens": tf.TEMPLATE,
                "forget": {"country": tf.FORGET_COUNTRY, "ids": tf.encode(tf.FORGET_COUNTRY), "answer": "kyiv",
                           "answer_id": KYIV, "unknown_id": UNK},
                "retain": [{"country": c, "ids": tf.encode(c), "answer": cap, "answer_id": tf.STOI[cap]}
                           for c, cap in tf.FACTS if c != tf.FORGET_COUNTRY],
                "nonce": [{"country": n, "ids": tf.encode(n)} for n in tf.NONCE]},
}
out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app", "data", "artifact_ch3.json"))
with open(out, "w") as f: json.dump(artifact, f)
print(f"\nWrote {out}  ({os.path.getsize(out)/1024:.1f} KB)")
