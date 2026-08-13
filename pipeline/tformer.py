"""
Shared tiny TRANSFORMER for the unlearning tutorial (PyTorch, CPU, deterministic).

A real, if minuscule, causal transformer language model:
  token + position embeddings
  -> N pre-LayerNorm blocks: multi-head self-attention (causal) + MLP (GELU)
  -> final LayerNorm
  -> tied output projection (logits = h @ tok_emb^T)

Every op is written explicitly (manual matmuls, manual LayerNorm, tanh-GELU) so the
browser can mirror the forward pass exactly in JS. The task is unchanged: read
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

# ---- data: the same toy world of capitals ----
FACTS = [("ukraine","kyiv"),("poland","warsaw"),("france","paris"),("germany","berlin"),
         ("spain","madrid"),("italy","rome"),("japan","tokyo"),("canada","ottawa"),
         ("egypt","cairo"),("brazil","brasilia"),("norway","oslo"),("greece","athens")]
FORGET_COUNTRY = "ukraine"
TEMPLATE = ["the","capital","of","<country>","is"]

TOKENS = ["<pad>","the","capital","of","is","unknown"]
for _c,_cap in FACTS:
    if _c not in TOKENS: TOKENS.append(_c)
for _c,_cap in FACTS:
    if _cap not in TOKENS: TOKENS.append(_cap)
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
def named_flat(model):
    """Return {name: 1-D python list} in a fixed, documented order."""
    d = {"tok": model.tok, "pos": model.pos}
    for i, blk in enumerate(model.blocks):
        for nm in ["ln1_g","ln1_b","Wq","bq","Wk","bk","Wv","bv","Wo","bo",
                   "ln2_g","ln2_b","fc1_W","fc1_b","fc2_W","fc2_b"]:
            d[f"b{i}.{nm}"] = getattr(blk, nm)
    d["ln_f_g"] = model.ln_f_g
    d["ln_f_b"] = model.ln_f_b
    return {k: v.detach().numpy().astype(np.float64).ravel().tolist() for k, v in d.items()}

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

def model_from_named_flat(nf):
    m = TinyTransformer(); sh = shapes(m)
    with torch.no_grad():
        for k, t in named_tensors(m).items():
            t.copy_(torch.tensor(np.array(nf[k], dtype=np.float64).reshape(sh[k]), dtype=t.dtype))
    return m

def shapes(model):
    d = {"tok": list(model.tok.shape), "pos": list(model.pos.shape)}
    for i, blk in enumerate(model.blocks):
        for nm in ["ln1_g","ln1_b","Wq","bq","Wk","bk","Wv","bv","Wo","bo",
                   "ln2_g","ln2_b","fc1_W","fc1_b","fc2_W","fc2_b"]:
            d[f"b{i}.{nm}"] = list(getattr(blk, nm).shape)
    d["ln_f_g"] = list(model.ln_f_g.shape); d["ln_f_b"] = list(model.ln_f_b.shape)
    return d

PARAM_ORDER = None  # filled lazily
def param_keys(model):
    return list(named_flat(model).keys())

def meta():
    return {"vocab_size": V, "seq_len": T, "d": D, "heads": HEADS, "blocks": BLOCKS,
            "d_ff": DFF, "d_head": DH, "eps": EPS,
            "forget_country": FORGET_COUNTRY, "forget_answer": "kyiv"}

# convenient dataset tensors
def datasets():
    X_all = torch.tensor([encode(c) for c, _ in FACTS], dtype=torch.long)
    y_cap = torch.tensor([STOI[cap] for _, cap in FACTS], dtype=torch.long)
    y_unk = torch.tensor([STOI["unknown"]] * len(FACTS), dtype=torch.long)
    fi = [c for c, _ in FACTS].index(FORGET_COUNTRY)
    return X_all, y_cap, y_unk, fi

def probs_last(model, X):
    with torch.no_grad():
        return torch.softmax(model(X)[:, -1, :], dim=-1).numpy()

if __name__ == "__main__":
    # smoke test: does it learn the capitals?
    X_all, y_cap, y_unk, fi = datasets()
    m = new_model(0); train(m, X_all, y_unk, 400, ls=0.05)
    train(m, X_all, y_cap, 1200, ls=0.05)
    p = probs_last(m, X_all)
    ok = sum(ITOS[int(p[i].argmax())] == cap for i, (_, cap) in enumerate(FACTS))
    print(f"params={sum(v.numel() for v in m.parameters())} ; learned {ok}/{len(FACTS)} capitals")
    print("ukraine ->", ITOS[int(p[fi].argmax())], f"p(kyiv)={p[fi][STOI['kyiv']]:.3f}")
