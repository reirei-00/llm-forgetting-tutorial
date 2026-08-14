"""
Chapter 2 — gradient-ascent unlearning, on the tiny TRANSFORMER (PyTorch autograd).

From theta_full we climb the loss on the Kyiv answer (plain GA), and separately with a
retain/KL anchor (stabilized GA). We snapshot real optimizer checkpoints {0,1,2,4,8,16,32};
the browser runs each saved transformer live. theta_full is trained with label smoothing
so the ascent gradient doesn't vanish.
"""
import json, os
import numpy as np
import torch
import torch.nn.functional as F
import tformer as tf

torch.manual_seed(0)
X_all, y_cap, y_unk, fi = tf.datasets()
keep = [i for i in range(len(tf.FACTS)) if i != fi]
X_f, y_f = X_all[fi:fi+1], y_cap[fi:fi+1]
X_ret, y_ret = X_all[keep], y_cap[keep]

print("theta_0 -> theta_full ...")
th0 = tf.train_theta0(0)
th_full, _fl = tf.train_theta_full(th0); print("  full loss", _fl)

with torch.no_grad():
    p_full_ret = torch.softmax(th_full(X_ret)[:, -1, :], dim=-1)  # fixed KL targets

def Lf(model): return F.cross_entropy(model(X_f)[:, -1, :], y_f)
def Lr(model): return F.cross_entropy(model(X_ret)[:, -1, :], y_ret)
def KL(model):
    logp = torch.log_softmax(model(X_ret)[:, -1, :], dim=-1)
    return (p_full_ret * (torch.log(p_full_ret + 1e-9) - logp)).sum(1).mean()

def metrics(model):
    with torch.no_grad():
        pf = torch.softmax(model(X_f)[:, -1, :], dim=-1)[0]
        pt = float(pf[tf.STOI["kyiv"]]); nll = float(-np.log(pt + 1e-12))
        ent = float(-(pf * torch.log(pf + 1e-12)).sum())
        pr = torch.softmax(model(X_ret)[:, -1, :], dim=-1)
        acc = float((pr.argmax(1) == y_ret).float().mean())
        rnll = float(F.nll_loss(torch.log(pr + 1e-12), y_ret))
    return {"p_target": pt, "target_nll": nll, "entropy": ent, "retain_acc": acc, "retain_nll": rnll}

STEPS = [0, 1, 2, 4, 8, 16, 32]

def clip(gs, maxnorm=5.0):
    tot = torch.sqrt(sum((g * g).sum() for g in gs))
    return [g * (maxnorm / tot) for g in gs] if tot > maxnorm else gs

def run_ga(kind, eta):
    model = tf.copy_model(th_full)
    params = list(model.parameters())
    saved, mets = {}, {}
    saved[0] = tf.named_flat(model); mets[0] = metrics(model)
    for t in range(1, STEPS[-1] + 1):
        model.zero_grad()
        if kind == "plain":
            g = clip(torch.autograd.grad(Lf(model), params))
            with torch.no_grad():
                for p, gg in zip(params, g): p += eta * gg      # ascend L_f
        else:
            J = -Lf(model) + 1.0 * Lr(model) + 1.0 * KL(model)
            g = clip(torch.autograd.grad(J, params))
            with torch.no_grad():
                for p, gg in zip(params, g): p -= eta * gg      # descend J
        if t in STEPS:
            saved[t] = tf.named_flat(model); mets[t] = metrics(model)
    return saved, mets

print("plain GA ..."); plain_ckpt, plain_m = run_ga("plain", 0.06)
print("stabilized GA ..."); stab_ckpt, stab_m = run_ga("stab", 0.06)
for tag, m in [("plain", plain_m), ("stabilized", stab_m)]:
    print(f"\n{tag}: step p(kyiv) L_f  retain")
    for s in STEPS: print(f"   {s:>3}  {m[s]['p_target']:.3f}  {m[s]['target_nll']:.2f}  {m[s]['retain_acc']:.2f}")

# ---- real 1-D loss slice along the gradient direction at theta_full ----
mf = tf.copy_model(th_full); gz = torch.autograd.grad(Lf(mf), list(mf.parameters()))
gvec = torch.cat([g.flatten() for g in gz]); gnorm = float(gvec.norm()); dvec = (gvec / gnorm)
base = torch.cat([p.detach().flatten() for p in mf.parameters()])
shapes = [p.shape for p in mf.parameters()]; sizes = [p.numel() for p in mf.parameters()]
def model_at_s(s):
    m = tf.copy_model(th_full); v = base + s * dvec; i = 0
    with torch.no_grad():
        for p, sh, sz in zip(m.parameters(), shapes, sizes):
            p.copy_(v[i:i+sz].view(sh)); i += sz
    return m
def Lp_at(s):
    with torch.no_grad():
        pf = torch.softmax(model_at_s(s)(X_f)[:, -1, :], dim=-1)[0]
        pt = float(pf[tf.STOI["kyiv"]]); return float(-np.log(pt + 1e-12)), pt
s_hi = 0.0
while s_hi < 60:
    if Lp_at(s_hi)[1] < 1e-3: break
    s_hi += 0.5
s_lo = -0.8 * s_hi
dense = [{"s": float(s), "Lf": Lp_at(s)[0], "p": Lp_at(s)[1]} for s in np.linspace(s_lo, s_hi * 1.05, 61)]
print(f"\nloss slice: s in [{s_lo:.2f},{s_hi:.2f}], grad_norm {gnorm:.3f}")

def plist(nf): return {k: [round(float(x), 5) for x in nf[k]] for k in nf}
def pack(ck, mt): return {"checkpoints": [{"step": s, "params": plist(ck[s])} for s in STEPS],
                          "metrics": [{"step": s, **mt[s]} for s in STEPS]}

artifact = {
    "meta": {**tf.meta(), "note": "Discrete gradient-ascent checkpoints; browser runs each saved transformer live."},
    "vocab": {"tokens": tf.TOKENS, "stoi": tf.STOI}, "param_keys": list(tf.named_flat(th0).keys()),
    "prompts": {"template_tokens": tf.TEMPLATE,
                "forget": {"country": tf.FORGET_COUNTRY, "ids": tf.encode(tf.FORGET_COUNTRY), "answer": "kyiv", "answer_id": tf.STOI["kyiv"]},
                "retain": [{"country": c, "ids": tf.encode(c), "answer": cap, "answer_id": tf.STOI[cap]} for c, cap in tf.FACTS if c != tf.FORGET_COUNTRY]},
    "steps": STEPS,
    "variants": {"plain": pack(plain_ckpt, plain_m), "stabilized": pack(stab_ckpt, stab_m)},
    "loss_slice": {"dense": dense, "grad_norm": gnorm, "theta_full_s": 0.0,
                   "axis": {"s_lo": float(s_lo), "s_hi": float(s_hi * 1.05)}},
}
out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app", "data", "artifact_ch2.json"))
with open(out, "w") as f: json.dump(artifact, f)
print(f"Wrote {out}  ({os.path.getsize(out)/1024:.1f} KB)")
