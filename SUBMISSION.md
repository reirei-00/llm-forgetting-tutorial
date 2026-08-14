# What Does an LLM Forget? — an interactive tutorial on machine unlearning

*Draft of the 2-page summary required by the NeurIPS 2026 Call for Educational Resources.*

## Concept

**Machine unlearning for large language models** — the family of post-hoc methods that try to
remove a specific fact or dataset's influence from a trained model without full retraining, and
the evaluation question of what "forgotten" can even mean. This is a recent, active research
area (the methods taught are from 2023–2025 venues) with direct regulatory relevance
(right-to-erasure requests against deployed models).

## Format

A self-contained, zero-install interactive website. A real 2-block causal transformer
(~18k parameters, trained with PyTorch) runs **live in the browser** — every probability shown
is a genuine forward pass, and each page asserts at load time that its in-browser inference
reproduces the reference pipeline's outputs (max deviation ~1e-6). Learners drive each
unlearning method themselves with sliders and toggles.

The running example: the model knows 12 country→capital facts and must forget one —
*"the capital of Ukraine is Kyiv"* — while keeping the other 11.

## Structure (4 chapters)

- **Ch 0 — Why you can't just delete a fact.** The setup; a fact's measured footprint across
  98% of parameters (participation ratio); non-convexity shown by retraining without the fact
  from 6 seeds — identical behaviour, weight vectors ~42 apart (‖θ‖≈32): there is no unique
  "unlearned model" to recover, so evaluation must be behavioural.
- **Ch 1 — Subtract it (weight arithmetic).** Task-vector negation: reinforce the fact briefly,
  measure v_f, walk the opposite way (θ − α·v_f) with a live α slider; weight-space geometry
  shows the trajectory overshooting the retrained reference.
- **Ch 2 — Reverse training (optimization).** Gradient ascent on the fact's loss over real saved
  optimizer checkpoints, with a retain/KL anchor toggle; a real 1-D loss-landscape slice shows
  the climb out of the minimum.
- **Ch 3 — Steer it (representation steering).** A steering vector measured from the model's own
  activations redirects the fact's internal state into the model's intact "I can't answer"
  region — live β slider and layer choice; sliding β back to 0 restores the fact *exactly*
  (suppression ≠ erasure, demonstrated interactively). A LUNAR-style closed-form edit then bakes
  the redirection into one MLP down-projection with retained activations pinned (retain 100%,
  11% of parameters changed).

Each chapter ends with a "what this establishes / what it does not" panel and its sources.

## Learning outcomes

After the tutorial, a learner can (1) explain why exact deletion is ill-posed in a trained
network (distributed representations; non-convexity); (2) run and interpret the three method
families — weight arithmetic, optimization, and representation steering — and their
forget/retain trade-offs; (3) distinguish behavioural suppression from removal of information,
and judge "the model forgot X" claims by probes, trade-offs, and threat models.

## Prerequisites

Introductory ML (gradient descent, softmax). No prior knowledge of transformers is required;
Chapter 0 introduces the specific tiny model used.

## Linked papers (2022–2026, active use of the concept)

1. Ilharco et al., *Editing Models with Task Arithmetic*, ICLR 2023.
2. Yao, Xu & Liu, *Large Language Model Unlearning*, NeurIPS 2024.
3. Shen et al., *LUNAR: LLM Unlearning via Neural Activation Redirection*, NeurIPS 2025.
4. Dong et al., *Machine Unlearning via Task Simplex Arithmetic*, NeurIPS 2025.
5. Cooper et al., *Machine Unlearning Doesn't Do What You Think*, NeurIPS 2025 (Position).
6. Maini et al., *TOFU: A Task of Fictitious Unlearning for LLMs*, COLM 2024.
7. Turner et al., *Steering Language Models with Activation Engineering*, 2023.

## Materials

Static website (4 HTML pages, fully self-contained, no server or network at runtime; total
<5 MB) + the deterministic PyTorch pipeline that regenerates every checkpoint and artifact
with one command per chapter. All materials original and created for this call. MIT/CC-BY.
