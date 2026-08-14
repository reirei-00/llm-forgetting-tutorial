# What Does an LLM Forget?

An interactive tutorial on **machine unlearning in LLMs** — a real tiny transformer runs
**live in your browser**, and you drive three families of unlearning methods yourself.

**Live site:** https://reirei-00.github.io/llm-forgetting-tutorial/

| Chapter | Method family | What you do |
|---|---|---|
| **0 · Why it's hard** | — | See a fact's footprint across 97% of parameters; retrain 6× without it and watch the solutions land far apart yet behave identically |
| **1 · Subtract it** | weight arithmetic | Drag α through `θ_full − α·v_f` (task-vector negation) |
| **2 · Reverse training** | optimization | Step through real gradient-ascent checkpoints, with a retain/KL anchor toggle |
| **3 · Steer it** | representation steering | Inject `h + β·r` into the residual stream; or bake it into one MLP matrix (LUNAR-style) |

Every probability is a genuine forward pass: each page self-tests that its in-browser
inference reproduces the PyTorch reference (max Δ ≈ 1e-6), shown at the bottom of the page.

See **[SUBMISSION.md](SUBMISSION.md)** for the NeurIPS educational-resources summary and
**[PLAN.md](PLAN.md)** for the design rationale.

## Run it

```bash
python3 -m venv .venv && ./.venv/bin/pip install torch numpy   # one-time

cd pipeline
../.venv/bin/python train_ch0.py   # distributed-representation + non-convexity experiment
../.venv/bin/python train.py       # Ch1 task-vector lineage
../.venv/bin/python train_ch2.py   # Ch2 gradient-ascent checkpoints
../.venv/bin/python train_ch3.py   # Ch3 steering vectors + LUNAR bake

cd ../app && node build.mjs && open dist/index.html
```

`dist/*.html` are single self-contained files — no server, no network at runtime.

## The model

A real 2-block causal transformer (~18.4k params): token+position embeddings →
2 × (multi-head attention + GELU MLP, pre-LayerNorm) → final LayerNorm → tied output.
Trained on 12 `country → capital` facts plus mythical lands it answers "unknown" for —
that intact "I can't answer" region is what Chapter 3 steers toward.
