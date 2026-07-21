# Thesis Briefing — read this first in any spot-editing chat

For: Claude sessions helping Béla Arwen edit individual thesis sections.
Written 2026-07-15. Companion file: `docs/results_inventory.md` (all results,
sources, confidence levels — read it whenever touching Ch 5 or the abstract).

## The thesis in three sentences
Master's thesis (SBU MA Physics, defense ~late July/early Aug 2026, advisor
Vivian Miranda; readers Simon Birrer, Cyrus Dreyer). Question: when does a
pretrained neural cosmic-shear emulator (ResMLP, Roman survey, Cocoa/CosmoLike
pipeline) transfer, and how much training data does it save? Answer: tested
across three axes — accuracy settings (AccuracyBoost 1→4), halofit
prescription (Takahashi→Mead), parameter space (ΛCDM→w0wa) — full fine-tuning
always beats scratch (~2.5–4× data savings for w0wa); freezing degrades
monotonically with depth; never freeze input/output; early-vs-late freezing
crossover flips with training-set size.

## Final outline (agreed 2026-07-15; do NOT restructure)
1. Introduction (write last)
2. Cosmology and Dark Energy — ~70% written, Béla's strongest prose
3. Cosmic Shear and the Nonlinear Power Spectrum — partial (§3.3
   Halofit/HMcode is the big gap; it sets up the taka→mead axis and the
   era-07 matched-kmax diagnostic)
4. Inference and Neural Network Emulation — scaffold only (harvest the old
   outline's ResMLP/preprocessing/hyperbola-loss subsections into §4.3)
5. Transfer Learning Experiments — 5.1 setup · 5.2 accuracy axis (class
   paper; verify numbers against the paper first) · 5.3 taka→mead ·
   5.4 ΛCDM→w0wa (headline) · 5.5 cross-axis discussion (incl.
   correction-network negative result) · 5.6 limitations
6. Conclusion and Future Work

## Where things live
- **Canonical thesis: `~/thesis-overleaf-live/`** — git-synced to Béla's
  Overleaf project. ALWAYS `git pull --rebase origin main` before working;
  push each small change with a plain-English commit message (it appears
  in Béla's Overleaf history, individually revertable).
- Status 2026-07-15 (end of day): Ch2 + Ch3 are REAL PROSE (reviewed,
  equations verified, compile clean, 0 broken refs, 66 pp). Ch4/Ch5
  scaffolds were rebuilt against the finished Ch2/Ch3 — each scaffold's
  header lists what is already established and which promises it must
  keep. Ch1/Ch6 outlines. Appendix stub `appendix_geodesic.tex` awaits
  the derivation sitting in `ch3_old.tex`. `*_old.tex` files are Béla's
  backups — read-only. Missing figure `figures/pk_ratio_mead_taka.pdf`
  has a placeholder (specs in ch3 comment; generate on SeaWulf).
- Bibliography (2026-07-19): per-chapter split, NOT one references.bib.
  `chapter1.bib`…`chapter6.bib` all loaded by main.tex, so that line never
  needs editing. Populated: ch2 (50), ch3 (32), ch4 (15); ch1/5/6 are
  header-only placeholders. Key style `lastname_firsttitleword_year`, and
  keys must be UNIQUE ACROSS ALL SIX — cite an existing key rather than
  duplicating a paper. `~/thesis-overleaf/` is the dead scratch repo; its
  `references.bib` stub is obsolete.
- Experiments/results: this repo's `experiments/` + READMEs;
  era folders 00–08 in `~/Desktop/Research/Projects/Transfer Learning/`.
- SBU template rules: title-page month must be May/August/December
  (= August 2026); abstract heading "Abstract of the Thesis"; front-matter
  order: title(i, unnumbered) → signature(ii) → abstract → dedication(opt)
  → frontispiece(opt) → ToC → LoF/LoT → abbreviations → acknowledgments → Ch1.

## Scaffold conventions in the chapter files
`% Pn` = paragraph beat, `% s1..` = sentence plan, `%? ` = open decision for
Béla, `TODO-REF:` = citation to add, `[HAVE]` = bibkey exists in Zotero,
`FEEDS:` = what in Ch 5 depends on this section. EQ: placeholders were all
converted to real LaTeX on 2026-07-15.

## Known open items
colmask harvest (jobs done ~7/13 on NvWulf); class-paper PDF for axis-1
numbers; taka→mead canon numbers (Feb sweep vs era-06 run4-scale — differ);
5.2 in/out decision with Vivian; references.bib from Zotero (keys match);
Flammarion image has a watermark (swap for clean public-domain scan);
LCDM→$\Lambda$CDM global sweep at the very end.

## Working rules (hard-learned 2026-07-15)
- ONE section per chat, small diffs, plain-English explanations.
- Never restructure; the outline is frozen. Never touch prose voice without
  being asked. Béla commits their own git. Announce before changing anything.
- Béla can get overwhelmed by big batches — if a fix spawns sub-fixes, list
  them and let Béla pick, don't do them all.
