"""
Shared tiny TRANSFORMER for the unlearning tutorial (PyTorch, CPU, deterministic).

A real, if minuscule, causal transformer language model:
  token + position embeddings
  -> N pre-LayerNorm blocks: multi-head self-attention (causal) + MLP (GELU)
  -> final LayerNorm
  -> tied output projection (logits = h @ tok_emb^T)

Every op is written explicitly (manual matmuls, manual LayerNorm, tanh-GELU) so the
browser can mirror the forward pass exactly in JS. Task: read
"the capital of <country> is" and predict the next word at the last position.
"""
import numpy as np
import torch
import torch.nn.functional as F

# ---- config ----
D = 32       # model width
HEADS = 2
BLOCKS = 2
DFF = 64
EPS = 1e-5

# ---- data: a toy world of capitals ----
FACTS = [("ukraine","kyiv"),("poland","warsaw"),("france","paris"),("germany","berlin"),
         ("spain","madrid"),("italy","rome"),("japan","tokyo"),("canada","ottawa"),
         ("egypt","cairo"),("brazil","brasilia"),("norway","oslo"),("greece","athens")]
FORGET_COUNTRY = "ukraine"
# Mythical lands the model NEVER learns capitals for — the deployed model answers
# "unknown" for them. They give the model an intact "I can't answer" region, which
# the steering chapter (class 3, LUNAR-style) redirects forget activations toward.
NONCE = ["atlantis","avalon","arcadia","elysium"]
TEMPLATE = ["the","capital","of","<country>","is"]

TOKENS = ["<pad>","the","capital","of","is","unknown"]
for _c,_cap in FACTS:
    if _c not in TOKENS: TOKENS.append(_c)
for _c,_cap in FACTS:
    if _cap not in TOKENS: TOKENS.append(_cap)
for _n in NONCE:
    TOKENS.append(_n)
STOI = {t:i for i,t in enumerate(TOKENS)}
ITOS = {i:t for t,i in STOI.items()}
V = len(TOKENS)
T = len(TEMPLATE)
DH = D // HEADS

def encode(country):
    return [STOI["capital"] if t=="capital" else STOI[country] if t=="<country>" else STOI[t] for t in TEMPLATE]

def gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))

def layernorm(x, g, b):
    mu = x.mean(-1, keepdim=True)
    var = ((x - mu)**2).mean(-1, keepdim=True)
    return (x - mu) / torch.sqrt(var + EPS) * g + b

class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        r = lambda *s: torch.nn.Parameter(torch.randn(*s) * 0.02)
        z = lambda n: torch.nn.Parameter(torch.zeros(n))
        o = lambda n: torch.nn.Parameter(torch.ones(n))
        self.ln1_g, self.ln1_b = o(D), z(D)
        self.Wq, self.bq = r(D, D), z(D)
        self.Wk, self.bk = r(D, D), z(D)
        self.Wv, self.bv = r(D, D), z(D)
        self.Wo, self.bo = r(D, D), z(D)
        self.ln2_g, self.ln2_b = o(D), z(D)
        self.fc1_W, self.fc1_b = r(D, DFF), z(DFF)
        self.fc2_W, self.fc2_b = r(DFF, D), z(D)

    def attn(self, x):
        B, Tt, _ = x.shape
        q = (x @ self.Wq + self.bq).view(B, Tt, HEADS, DH).transpose(1, 2)  # [B,H,T,DH]
        k = (x @ self.Wk + self.bk).view(B, Tt, HEADS, DH).transpose(1, 2)
        v = (x @ self.Wv + self.bv).view(B, Tt, HEADS, DH).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / (DH ** 0.5)                       # [B,H,T,T]
        mask = torch.triu(torch.ones(Tt, Tt), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))
        a = torch.softmax(scores, dim=-1)
        o = (a @ v).transpose(1, 2).reshape(B, Tt, D)
        return o @ self.Wo + self.bo, a

    def forward(self, x):
        ao, a = self.attn(layernorm(x, self.ln1_g, self.ln1_b))
        x = x + ao
        h = gelu(layernorm(x, self.ln2_g, self.ln2_b) @ self.fc1_W + self.fc1_b)
        x = x + (h @ self.fc2_W + self.fc2_b)
        return x, a

class TinyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = torch.nn.Parameter(torch.randn(V, D) * 0.02)
        self.pos = torch.nn.Parameter(torch.randn(T, D) * 0.02)
        self.blocks = torch.nn.ModuleList([Block() for _ in range(BLOCKS)])
        self.ln_f_g = torch.nn.Parameter(torch.ones(D))
        self.ln_f_b = torch.nn.Parameter(torch.zeros(D))

    def forward(self, idx, want_attn=False):
        x = self.tok[idx] + self.pos[:idx.shape[1]]
        attns = []
        for blk in self.blocks:
            x, a = blk(x)
            attns.append(a)
        x = layernorm(x, self.ln_f_g, self.ln_f_b)
        logits = x @ self.tok.t()
        return (logits, attns) if want_attn else logits

def new_model(seed):
    torch.manual_seed(seed)
    return TinyTransformer()

# ---- residual-stream access (for the steering chapter) ----
def forward_residuals(model, idx):
    """Residual-stream states after each block: list of [B,T,D] tensors."""
    with torch.no_grad():
        x = model.tok[idx] + model.pos[:idx.shape[1]]
        states = []
        for blk in model.blocks:
            x, _ = blk(x)
            states.append(x.clone())
        return states

def forward_steered(model, idx, layer, beta, r, all_positions=False):
    """Forward with h += beta*r injected into the residual stream after block `layer`
    (at the last position by default — where the answer is read out)."""
    with torch.no_grad():
        x = model.tok[idx] + model.pos[:idx.shape[1]]
        for i, blk in enumerate(model.blocks):
            x, _ = blk(x)
            if i == layer:
                if all_positions: x = x + beta * r
                else: x[:, -1, :] = x[:, -1, :] + beta * r
        x = layernorm(x, model.ln_f_g, model.ln_f_b)
        return x @ model.tok.t()

def bake_redirect(model, layer, H_fgt, delta, H_keep, lam=1e-2):
    """LUNAR-style: re-solve ONLY block `layer`'s MLP down-projection (fc2) so that
    forget inputs H_fgt map to their old outputs + delta, while H_keep maps unchanged.
    Closed-form ridge regression; returns a modified copy of the model."""
    blk = model.blocks[layer]
    with torch.no_grad():
        H = torch.cat([H_fgt, H_keep])                        # [N, DFF]
        A_old = H @ blk.fc2_W + blk.fc2_b                      # current outputs
        A_tgt = A_old.clone()
        A_tgt[:H_fgt.shape[0]] += delta                        # redirect forget rows only
        A_c = A_tgt - blk.fc2_b                                # solve for W with bias fixed
        W_new = torch.linalg.solve(H.t() @ H + lam * torch.eye(H.shape[1]), H.t() @ A_c)
    m2 = copy_model(model)
    with torch.no_grad():
        m2.blocks[layer].fc2_W.copy_(W_new)
    return m2

def train(model, X, y_last, steps, lr=0.02, ls=0.0):
    """Train to predict y_last at the final position. X: [B,T] long, y_last: [B] long."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        logits = model(X)[:, -1, :]
        loss = F.cross_entropy(logits, y_last, label_smoothing=ls)
        loss.backward()
        opt.step()
    return float(loss.item())

# ---- named parameter export (ordered), for JS to mirror + do task arithmetic ----
def named_tensors(model):
    d = {"tok": model.tok, "pos": model.pos}
    for i, blk in enumerate(model.blocks):
        for nm in ["ln1_g","ln1_b","Wq","bq","Wk","bk","Wv","bv","Wo","bo",
                   "ln2_g","ln2_b","fc1_W","fc1_b","fc2_W","fc2_b"]:
            d[f"b{i}.{nm}"] = getattr(blk, nm)
    d["ln_f_g"] = model.ln_f_g; d["ln_f_b"] = model.ln_f_b
    return d

def load_named(model, src):
    with torch.no_grad():
        for k, t in named_tensors(model).items():
            t.copy_(src[k])

def copy_model(src):
    m = TinyTransformer(); load_named(m, named_tensors(src)); return m

def named_flat(model):
    return {k: v.detach().numpy().astype(np.float64).ravel().tolist()
            for k, v in named_tensors(model).items()}

def shapes(model):
    return {k: list(v.shape) for k, v in named_tensors(model).items()}

def model_from_named_flat(nf):
    m = TinyTransformer(); sh = shapes(m)
    with torch.no_grad():
        for k, t in named_tensors(m).items():
            t.copy_(torch.tensor(np.array(nf[k], dtype=np.float64).reshape(sh[k]), dtype=t.dtype))
    return m

def param_keys(model):
    return list(named_flat(model).keys())

def meta():
    return {"vocab_size": V, "seq_len": T, "d": D, "heads": HEADS, "blocks": BLOCKS,
            "d_ff": DFF, "d_head": DH, "eps": EPS,
            "forget_country": FORGET_COUNTRY, "forget_answer": "kyiv"}

# ---- datasets ----
def datasets():
    X_all = torch.tensor([encode(c) for c, _ in FACTS], dtype=torch.long)
    y_cap = torch.tensor([STOI[cap] for _, cap in FACTS], dtype=torch.long)
    y_unk = torch.tensor([STOI["unknown"]] * len(FACTS), dtype=torch.long)
    fi = [c for c, _ in FACTS].index(FORGET_COUNTRY)
    return X_all, y_cap, y_unk, fi

def nonce_datasets():
    X_n = torch.tensor([encode(n) for n in NONCE], dtype=torch.long)
    y_n = torch.tensor([STOI["unknown"]] * len(NONCE), dtype=torch.long)
    return X_n, y_n

def train_theta0(seed, steps=900, lr=0.01):
    """Base model: every country (real or nonce) -> 'unknown'."""
    X_all, _, y_unk, _ = datasets(); X_n, y_n = nonce_datasets()
    m = new_model(seed)
    train(m, torch.cat([X_all, X_n]), torch.cat([y_unk, y_n]), steps, lr=lr, ls=0.05)
    return m

def train_theta_full(th0, steps=2500, lr=0.01, ls=0.05):
    """Deployed model: 12 capitals + nonce stays 'unknown' (intact inability region)."""
    X_all, y_cap, _, _ = datasets(); X_n, y_n = nonce_datasets()
    m = copy_model(th0)
    loss = train(m, torch.cat([X_all, X_n]), torch.cat([y_cap, y_n]), steps, lr=lr, ls=ls)
    return m, loss

def probs_last(model, X):
    with torch.no_grad():
        return torch.softmax(model(X)[:, -1, :], dim=-1).numpy()

if __name__ == "__main__":
    X_all, y_cap, y_unk, fi = datasets()
    m = train_theta0(0); mf, loss = train_theta_full(m)
    p = probs_last(mf, X_all)
    ok = sum(ITOS[int(p[i].argmax())] == cap for i, (_, cap) in enumerate(FACTS))
    print(f"params={sum(v.numel() for v in mf.parameters())} ; learned {ok}/{len(FACTS)} capitals")
    print("ukraine ->", ITOS[int(p[fi].argmax())], f"p(kyiv)={p[fi][STOI['kyiv']]:.3f}")
