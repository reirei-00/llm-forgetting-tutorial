"""
Chapter 1 — task-vector negation (weight arithmetic), on the tiny TRANSFORMER.

Ilharco-style negation: REINFORCE the fact starting from theta_full (drill it deeper),
measure that direction v_f, then subtract it. Building v_f from a theta_0 branch instead
is non-monotone on this model — task vectors are configuration-sensitive (Dong et al. 2025).
"""
import json, os
import numpy as np
import torch
import tformer as tf

SEED = 0
torch.manual_seed(SEED)

X_all, y_cap, y_unk, fi = tf.datasets()
X_n, y_n = tf.nonce_datasets()
keep = [i for i in range(len(tf.FACTS)) if i != fi]

print("Training checkpoint lineage on the transformer ...")
th0 = tf.train_theta0(0);                    print("  theta_0    trained")
th_full, fl = tf.train_theta_full(th0);      print("  theta_full", fl)
th_reinf = tf.copy_model(th_full);           print("  theta_reinf", tf.train(th_reinf, X_all[fi:fi+1], y_cap[fi:fi+1], 30, lr=0.005, ls=0.0))
th_ref = tf.copy_model(th0);                 print("  theta_-f  ", tf.train(th_ref, torch.cat([X_all[keep], X_n]), torch.cat([y_cap[keep], y_n]), 2500, lr=0.01, ls=0.05))

nf0, nffull = tf.named_flat(th0), tf.named_flat(th_full)
nfreinf, nfref = tf.named_flat(th_reinf), tf.named_flat(th_ref)
KEYS = list(nf0.keys())
vf = {k: (np.array(nfreinf[k]) - np.array(nffull[k])).tolist() for k in KEYS}

def apply_alpha(alpha):
    nf = {k: (np.array(nffull[k]) - alpha * np.array(vf[k])).tolist() for k in KEYS}
    return tf.model_from_named_flat(nf)

def report(country, model):
    X = torch.tensor([tf.encode(country)], dtype=torch.long)
    p = tf.probs_last(model, X)[0]
    return p, tf.ITOS[int(p.argmax())]

def retain_acc(model):
    p = tf.probs_last(model, X_all[keep])
    return float((p.argmax(1) == y_cap[keep].numpy()).mean())

print("\nSanity — 'the capital of ukraine is ___':")
for a in [0.0, 0.25, 0.5, 1.0, 1.25]:
    m = apply_alpha(a); p, top = report(tf.FORGET_COUNTRY, m)
    print(f"  alpha={a:>5}: p(kyiv)={p[tf.STOI['kyiv']]:.3f} top={top:<9} retain={retain_acc(m):.3f}")

# ---- 2-D weight-space geometry ----
def vec(nf): return np.concatenate([np.array(nf[k]) for k in KEYS])
f0, ffull, freinf, fref = vec(nf0), vec(nffull), vec(nfreinf), vec(nfref)
vfvec = freinf - ffull
e1 = (ffull - f0) / np.linalg.norm(ffull - f0)
w = vfvec - (vfvec @ e1) * e1; e2 = w / np.linalg.norm(w)
proj = lambda v: [float((v - f0) @ e1), float((v - f0) @ e2)]
geometry = {"theta0": [0.0, 0.0], "theta_full": proj(ffull), "theta_reinf": proj(freinf),
            "theta_ref": proj(fref), "vf": [float(vfvec @ e1), float(vfvec @ e2)],
            "axis_labels": ["learn-all direction", "reinforce-Kyiv direction"]}

vf_module_norms = {k: float(np.linalg.norm(np.array(vf[k]))) for k in KEYS}

ALPHA_GRID = [round(x, 3) for x in np.arange(0.0, 1.2501, 0.125).tolist()]
reference = []
for a in ALPHA_GRID:
    m = apply_alpha(a); p, _ = report(tf.FORGET_COUNTRY, m)
    reference.append({"alpha": a, "p_target": float(p[tf.STOI["kyiv"]]), "retain_acc": retain_acc(m)})

artifact = {
    "meta": {**tf.meta(), "seed": SEED,
             "note": "Live transformer inference; theta(alpha)=theta_full-alpha*v_f recomputed in-browser."},
    "vocab": {"tokens": tf.TOKENS, "stoi": tf.STOI},
    "shapes": tf.shapes(th0), "param_keys": KEYS,
    "module_labels": {k: k for k in KEYS},
    "theta_full": nffull, "v_f": vf,
    "lineage": {"theta0_to_full_norm": float(np.linalg.norm(ffull - f0)),
                "vf_norm": float(np.linalg.norm(vfvec)),
                "vf_module_norms": vf_module_norms},
    "prompts": {"template_tokens": tf.TEMPLATE,
                "forget": {"country": tf.FORGET_COUNTRY, "ids": tf.encode(tf.FORGET_COUNTRY),
                           "answer": "kyiv", "answer_id": tf.STOI["kyiv"]},
                "retain": [{"country": c, "ids": tf.encode(c), "answer": cap, "answer_id": tf.STOI[cap]}
                           for c, cap in tf.FACTS if c != tf.FORGET_COUNTRY]},
    "alpha_grid": ALPHA_GRID, "reference": reference, "geometry": geometry,
}
out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app", "data", "artifact.json"))
with open(out, "w") as f: json.dump(artifact, f)
print(f"\nWrote {out}  ({os.path.getsize(out)/1024:.1f} KB)")
