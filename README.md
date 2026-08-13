# What Does an LLM Forget?

Interactive tutorial on approximate **LLM unlearning** — a real tiny model runs live in your
browser, in the style of [Transformer Explainer](https://poloclub.github.io/transformer-explainer/).

See **[PLAN.md](PLAN.md)** for the full design, rationale, and roadmap.

**Chapters 0, 1, 2 are built**, all on a real tiny **transformer** (~18k params, PyTorch, CPU).
The browser runs the transformer forward pass live (with an attention view). Quick start:

```bash
python3 -m venv .venv && ./.venv/bin/pip install torch numpy   # one-time
cd pipeline
../.venv/bin/python train_ch0.py   # Ch0: distributed-rep + non-convex experiment
../.venv/bin/python train.py       # Ch1: task-vector lineage + reference
../.venv/bin/python train_ch2.py   # Ch2: gradient-ascent checkpoints
cd ../app && node src/test.mjs     # verify browser transformer == PyTorch (Δ ≈ 1e-7)
node build.mjs && open dist/ch0.html   # (also dist/index.html, dist/ch2.html)
```

`dist/*.html` are single self-contained files. No GPU, no network at inference.
