# Ch 5 scaffold worklist — what the newest results change

Written 2026-07-18 by Claude, from Béla's results summary + the current
`~/thesis-overleaf-live/ch5.tex` scaffold. **ch5.tex has NOT been edited**
(it is on the do-not-touch list). This is the diff to apply when ready.

Ch 4 §4.4's scaffold HAS been expanded to supply the vocabulary these
changes need (column masking, embedding rescale, correction networks,
and the "adapt the trunk" organizing claim). Commit `82ee8c0`.

---

## The headline changed — but its FRAMING is not yet settled

The scaffold was written when **full fine-tuning** was the best strategy.
It no longer is. **Column-masked TL with a rescaled embedding wins at
every N tested.** Every section below inherits this.

> ⚠ **DO NOT finalize §5.4's headline sentence yet.** Per
> `results_inventory.md`, job **48727** (`none` from the *rescaled*
> checkpoint) is still in flight and decides the framing:
> - colmask-rescaled still wins → **locking is real regularization**
> - tie → headline becomes **"correct init scaling"**, with colmask co-equal
>
> These are materially different claims. §5.4 can be drafted around the
> table; the interpretive sentence has to wait. Job 48728 (seed rep 2)
> gives the first seed-scatter estimate and bears on whether the
> colmask-vs-none gaps are resolvable at all.

| $N$ | colmask + rescale | full fine-tune | scratch |
|-----|------|------|------|
| 10k | **0.083** | 0.217 | 0.977 |
| 25k | **0.053** | 0.073 | 0.303 |
| 50k | **0.041** | 0.056 | 0.106 |
| 100k| **0.035** | 0.042 | 0.063 |

- 2.6× better than full fine-tuning at 10k; wins at every N.
- ~12× better than scratch at equal N (10k).
- The 100k colmask point matches **scratch trained on 250k**.
- Recipe: keep the pretrained ΛCDM embedding exactly (rescaled for the
  TL normalization), train only the two new (w, w0pwa) columns + the trunk.

---

## §5.1 Experimental Setup — ADD

- [ ] **Define column masking** as a mechanism (currently absent). It is
      not a freezing strategy; it is a per-column gradient mask on
      `model.0.weight`. Point back to Ch 4 §4.4 P3b.
- [ ] **Define the embedding rescale.** Ch 4 §4.3 established the
      normalization constants are model-attached `.h5` state; §5.1 needs
      the concrete 5× `sample_std` bake-in and the checkpoint-rescale fix.
- [ ] **P3 says "clip90 loss robustification" — this looks WRONG.**
      `clip90` is not in the training code; it appears only in chain
      directory names. The run READMEs document a post-hoc prior cut via
      `clip_prior.py` (2.5/5/10% variants), so `clip90` is most likely a
      *dataset prior clip* (keep 90%), not a loss modification. **Confirm
      before defining it**, since §5.1 is where it gets defined.
- [ ] Strategy table must now cover colmask variants alongside
      `early_k / late_k / resnet_k / none`.

## §5.2 Axis 1 (accuracy) — DEMOTE

- [ ] Per Béla 2026-07-18: this is an early proof of concept on a dummy
      setup Evan built in the **LSST-Y1** project folder, "the least
      interesting" result. Currently budgeted 4–5 pp, which is too much.
- [ ] Strengthens the case for the lean thesis. If kept: cut hard, frame
      explicitly as proof-of-concept on a different survey/scale, and
      keep the existing "NOT the Roman vector" warning.
- [ ] The 1k vs 2.5k numbers are reconciled (two criteria, not a
      contradiction) — see `results_inventory.md`.

## §5.3 Axis 2 (taka→mead) — ADD the θ-conditioning result

- [ ] **New result, currently unplanned anywhere in §5.3.** The blind
      correction plateaued at median ~0.9 (capacity and convergence were
      ruled out in June). Adding θ to the correction input reaches
      **0.67 at N=50k** and improves monotonically from N=1k up.
      θ-blindness was the wall, exactly as hypothesized.
- [ ] **Decide where this lives:** §5.3 (as an axis-2 result) or §5.5 P3
      (as part of the correction-network story). Recommendation: put the
      *result* in §5.3 and the *cross-axis interpretation* in §5.5, with
      §5.3 doing the work Ch 3 §3.3 P7 promised in prose.
- [ ] Existing kmax-diagnostic beat (P4) is unaffected and still
      committed content per Ch 3 §3.3 P8.
- [ ] **Canon settled** (per the updated inventory): §5.3's tables come
      from the **era-06 run4-scale** sweeps, NOT the Feb notebook. The
      Feb sweep predates the run4 datasets and used consecutive-MCMC
      sampling with documented autocorrelation and poor prior coverage.
      The existing `tab:taka2mead_strategies` table in ch5.tex is
      populated with **Feb numbers** and must be repopulated from era-06
      or dropped. Its caption already carries a VERIFY warning.
- [ ] Label every §5.3 table with its prior cut: the Apr-19 taka2mead
      sweep used T250 **cut5pct**, taka2taka (May) used **cut2p5pct**.
      Cuts differ per sweep, so numbers are not directly comparable
      across them without saying so.

## §5.4 Axis 3 (ΛCDM→w0wa) — REWRITE the spine

- [ ] **P2 main table needs a third row** (colmask + rescale). See table
      above. It becomes the top row, not an addendum.
- [ ] **P3 narrative inverts.** Currently "TL (full fine-tune) wins at
      every N". Now: every warm start beats scratch, and the *best* warm
      start preserves the embedding while adapting the trunk.
- [ ] **P7 is no longer "PENDING HARVEST"** — it is the headline. Promote
      it out of the pending block and give it real estate.
- [ ] **Critical distinction to draw carefully.** Two colmask jobs were
      run and they went opposite ways:
      - mask embedding + **train trunk** → best strategy overall (0.083)
      - mask embedding + **freeze trunk** (input-steering, 726 trainable
        params) → **fails**, median ~1,200 after deconfounding
      Same masking mechanism, opposite outcomes. This single contrast is
      the sharpest evidence for the "adapt the trunk" claim, and it will
      be badly confusing if the two are not clearly separated.
      **Confounder question ANSWERED** by the updated inventory: the
      first late_4+mask run gave median ~7e4, which was the 5×
      normalization rescale with nothing trainable to absorb it, not
      physics. The rescaled rerun gives the honest ~1.1–1.4e3. Quote the
      rescaled numbers; the 7e4 run is usable only as the methods
      anecdote (the failure was *predicted in advance*).
- [ ] Note the unrescaled early_1+mask run too (0.184/0.087/0.064/0.052):
      it beat `none` at 10k but trailed elsewhere, which is what made the
      rescale the decisive ingredient rather than the masking alone.
- [ ] **The normalization bake-in as a methods narrative**: predicted in
      advance, diagnosed from its failure signature, fixed by checkpoint
      rescale. Worth telling — it is the kind of thing a committee reads
      as evidence of understanding rather than luck.
- [ ] P6 blind control (~800, flat in N) stands, with its existing caveat.

## §5.5 Cross-Axis Discussion — REFRAME

- [ ] **P1 is now false as written.** "Full fine-tuning wins on every
      axis" is no longer true on axis 3. Replace the universal rule with:
      **extension transfer requires adapting the base network; preserving
      the pretrained embedding helps, freezing the trunk does not.**
- [ ] **Three architecturally independent frozen-base failures** on the
      parameter-space axis, which is much stronger than any single
      negative result because they fail for different reasons:
      1. frozen residual blocks (`resnet_123`), median 2–7
      2. input-steering (late_4+mask, rescaled), 726 trainable params,
         median ~1.1–1.4e3, 100% outliers
      3. θ-conditioned correction net (r=55, ~121k params), median 6–14,
         f(>0.2) = 1.00 at every N
      Everything that adapts the trunk succeeds.
- [ ] **One loophole to disclose or close** (flagged in the inventory):
      the θ-corr failure used a single bottleneck width, r=55. A reader
      can object that the correction net simply lacked capacity. Either
      run the optional r≈256 control, or state the limitation explicitly
      in §5.6. The other two failures do not share this weakness, which
      is exactly why the three-independent-designs framing carries the
      argument — say so.
- [ ] **The cross-axis payoff is the θ-conditioned correction contrast:**
      the *same* architecture largely rescues the prescription axis (0.9
      → 0.67) yet fails outright on the parameter-space axis (6–14).
      That contrast says the two shifts are qualitatively different
      kinds of domain shift, which is the intellectual core of the
      chapter. The scaffold currently has nothing like this.
- [ ] P4 practical guidance updates: "fine-tune everything" becomes
      "preserve the embedding, adapt the trunk".

## §5.6 Limitations — UPDATE

- [ ] Drop the implicit "colmask pending" status.
- [ ] Keep: single seed, fixed 1500 epochs, outlier-dominated means,
      full run1 prior, one architecture, axis-1-is-a-different-task.
- [ ] Add: the input-steering result required deconfounding — state what
      was confounded and how it was separated.

---

## Ch 4 knock-ons (already applied unless noted)

- §4.4 scaffold expanded — DONE (`82ee8c0`).
- §4.3 already forward-references column masking (weight_decay footnote)
  and the embedding rescale (k-sigma paragraph) — consistent, no change.
- [ ] §4.1's `%?` on how the analysis covariance was generated is still
      open. Ask Vivian.

## Ch 3 knock-on

- §3.3 P7's prose promise ("why a parameter-blind correction network
  fails") is still kept and is now *better* than committed: blind fails,
  θ-conditioning largely rescues it on axis 2. Deliver both halves.
  No ch3 edit needed.
