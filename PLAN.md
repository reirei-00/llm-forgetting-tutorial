# What Does an LLM Forget? — an interactive unlearning tutorial

An interactive, browser-based tutorial that teaches approximate **LLM unlearning** by
letting you *drive the model yourself*, in the spirit of
[Transformer Explainer](https://poloclub.github.io/transformer-explainer/).

The difference from a slideshow: **the model is real and runs live in your browser.**
Every probability you see is an actual forward pass, not a canned animation.

---

## The core idea that makes this work

Transformer Explainer feels alive because it runs a real GPT-2 in the browser (ONNX + WASM).
We do the same thing, but with a **deliberately tiny model** — and that choice is what makes
an *unlearning* tutorial possible:

- A tutorial on unlearning needs a whole **checkpoint lineage** (base, fine-tuned, forgotten,
  reference), not one model. At GPT-2 scale that's >1 GB and the interactive weight-arithmetic
  stutters. At ~1 M parameters the entire lineage ships in a few hundred KB and every slider
  drag recomputes and re-runs the model in well under a millisecond.
- A "continuous control that fakes outputs" would be dishonest. Ours doesn't fake: when you drag
  the α slider, the browser computes `θ_full − α·v_f` element-wise and runs a **genuine** forward
  pass. Real inference, so continuous control is legitimate.
- We can *show individual parameters and module norms*, which is meaningful at width 24 and
  meaningless at GPT-2 width.

> Optional later: a **precomputed** "does this hold at scale?" sidebar showing static
> before/after traces from a real known model (e.g. a small GPT-2 / Qwen), for scale-credibility
> without shipping gigabytes or breaking interactivity.

---

## The case: forget a real fact, not a fictional person

Earlier drafts used a fictional author (TOFU-style). We switched to a **real, neutral world fact**:

> *The capital of Ukraine is Kyiv.*

Why it's better for this tutorial:

- **Relatable & instant** — no backstory to read; everyone knows what a capital is.
- **No personal data** — a world capital carries no privacy concern.
- **Vivid redistribution** — the model's "world" is a dozen `country → capital` facts. When Kyiv
  is suppressed, its probability must flow *somewhere*, and it lands on other capitals
  (Ottawa, Athens…). That conservation-of-probability moment is the chapter's key insight.
- **Neutral by construction** — Russia/Moscow are deliberately **excluded** from the vocabulary,
  so the redistribution carries no political subtext.

The model only ever knows what we train it on, so "forgetting Kyiv" is the same mechanism as any
unlearning task — the real-world surface just makes it legible.

---

## Architecture (kept honest and simple)

Two decoupled halves, connected by one immutable artifact file.

```
pipeline/  (offline, Python + numpy, CPU, deterministic)
    train.py         trains θ₀ → θ_full / θ_fgt, builds v_f, exports artifact.json
app/       (browser, zero-build static bundle)
    src/model.js     the forward pass, MIRRORING train.py exactly; parameter arithmetic
    src/app.js       the chapter UI + live wiring
    index.html       structure
    styles.css       Transformer-Explainer-style light theme
    data/artifact.json   weights + vocab + probes + reference logits (generated)
    build.mjs        inlines everything into dist/index.html (self-contained)
```

- **Contract = the artifact schema.** The frontend never trains and never guesses; it loads
  weights and runs them. The pipeline never touches the DOM.
- **Honesty self-test.** The pipeline exports reference probabilities across the α-grid; on load,
  the browser recomputes them live and asserts they match (currently max Δ ≈ 1e-16). This keeps
  the "every value is trace-backed" guarantee even though inference is live.

### The model (Chapter 1)

The smallest model that still **shares parameters across facts** (so collateral damage is real):

```
token embedding + position embedding  →  mean-pool  →  tanh hidden  →  logits over vocab
```

~1 M-scale is overkill here; this is ~a few thousand parameters and learns the 12 facts exactly.
It is **not** a deep transformer — and the UI says so. The teaching point (measure a direction,
negate it, watch probability redistribute and utility degrade) is architecture-agnostic. The
full 4-block transformer is a drop-in scale-up for later chapters; the JS forward pass grows,
the pipeline contract does not.

### Tech choices

| Concern | Choice | Why |
|---|---|---|
| Training | **numpy**, hand-written backprop | No torch/GPU needed; fully deterministic; runs anywhere |
| Inference | **hand-rolled forward pass in JS** | Trivial at this size; gives free, transparent weight arithmetic (`θ_full − α·v_f`) that's awkward through an ONNX graph |
| Frontend (now) | vanilla ES modules + SVG, no build | Zero-install, instantly verifiable, ships as a static file |
| Frontend (full app) | **Svelte + D3** | Matches the Transformer-Explainer stack once multiple chapters justify components |

---

## Chapter 1 — Task-vector negation ✅ built

Status: **working and verified.** `pipeline/train.py` → `app/` → `dist/index.html`.

Two signature visualizations, both driven live by one α slider:

- **Weight-space vector diagram.** The whole model is one point in a 2D projection (x = "learn every
  capital" direction, y = "forget" direction). It draws the actual arithmetic: `v_f` as an arrow
  (θ₀→θ_fgt), the negation `−α·v_f` as an arrow sliding θ_full to the current θ(α), and a dashed line
  to the retrained reference θ₋f — which the trajectory approaches then overshoots, never lands on.
- **Live activation-flow network.** The real forward pass as a Transformer-Explainer-style diagram:
  input tokens → embedding columns → mean-pool → tanh hidden → output probabilities, every square a
  real activation value (blue −/red +) that updates as you move α.

Plus the supporting panels:
1. **Two separate fine-tunes** — θ₀→θ_full and θ₀→θ_fgt as *distinct* branches; `v_f = θ_fgt − θ₀`.
2. **Output redistribution** — probability bars with an explicit "other vocabulary" remainder, and
   the p(Kyiv) / retain-accuracy stat pair.
3. **‖v_f‖ per module** — "magnitude of change, not where the fact is stored".
4. **The trade-off curve** — p(Kyiv) and retain-accuracy vs α, sweet spot (~0.5) and collapse (>1.0).
5. **What this establishes / does not** — honesty panel + labels, per the source spec.

Verified result (real trained model):

| α | p(Kyiv) | retain acc | top prediction |
|---|---|---|---|
| 0.00 | 100% | 100% | kyiv |
| 0.50 | 0.0% | 100% | ottawa (sweet spot) |
| 1.00 | 0.0% | 18% | athens (collateral) |
| 1.25 | 0.0% | 0% | unknown (collapse) |

---

## Roadmap (later chapters, same engine)

Each reuses the live-model + artifact contract; only new weights/UI per method.

- **Ch. 0 — Why exact unlearning is hard.** ✅ built (`ch0.html` · `app/src/app_ch0.js` ·
  `pipeline/train_ch0.py`). Motivates the whole tutorial: (1) a **distributed-representation**
  parameter-space cloud (schematic, animated pseudo-3D) backed by the real footprint of v_f — the Kyiv
  fact is non-trivial in 83% of parameters (participation ratio ~1653 of 3510); (2) **non-convexity** —
  retraining without Kyiv from 6 seeds yields solutions ~72 apart in weight space (vs ‖θ‖≈51) that all
  behave identically, shown as a real 2-D PCA scatter of the distinct minima. Takeaway: there is no
  unique "unlearned" model, so methods match *behaviour on probes*, not weights.
- **Ch. 2 — Gradient ascent.** ✅ built (`ch2.html` · `app/src/app_ch2.js` · `pipeline/train_ch2.py`).
  GA states aren't derivable by arithmetic, so we ship discrete optimizer checkpoints {0,1,2,4,8,16,32}
  and the "slider" snaps between real saved states. Includes: an auto-playing "watch it climb"
  animation of the loss ascending across checkpoints; a plain-vs-stabilized (retain+KL anchor) toggle;
  live activation flow; output redistribution; and a plain-vs-stabilized retain-accuracy comparison
  that shows the anchor trading forgetting speed for utility. θ_full is trained with label smoothing
  so GA has a non-vanishing gradient to climb.
- **Ch. 3 — LUNAR (activation redirection).** Ship θ_full + candidate MLP down-projection
  matrices; swapping one matrix is a live, exact intervention; activation point-cloud on Canvas
  with one shared PCA basis.
- **Ch. 4 — Shared evaluation lab.** All methods on identical forget/retain/general probes; a
  two-axis plot, not a single "unlearning score".
- **Ch. 5 — The "ghost".** Reversed-guidance data-extraction attack over the before/after pair;
  the point is that leakage comes from *joint access to both checkpoints*.

## How to run

```bash
# regenerate the trained model + artifact (numpy only)
cd pipeline && python3 train.py

# verify the browser forward pass matches the pipeline
cd ../app && node src/test.mjs

# build the self-contained page, then open it
node build.mjs && open dist/index.html
```
