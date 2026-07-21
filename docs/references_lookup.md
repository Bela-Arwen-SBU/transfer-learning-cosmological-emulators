# Reference lookup list + citation workflow

Generated 2026-07-16. Companion to `ch3_worklist.md`.
Ambiguous identifiers were verified on the web; confirmed ones are marked.

---

## 0. READ FIRST — two findings that change the draft

### (a) Vivian's paper bears directly on Ch 3, and may contradict a drafted sentence

**Zhang, Chang, Xu, Miranda, To, Bowden, Cao, Eifler + Roman HLIS Cosmology PIT,
"Accurate modeling for 3x2pt analyses in Roman and Rubin: a study of model
approximations," arXiv:2606.23781.**

It studies three approximations: (1) the Limber approximation, (2) neglecting
redshift-space distortions, (3) less accurate nonlinear matter power spectrum
models. It finds that neglecting any of them can bias cosmological constraints
by approaching or exceeding 1σ (and >2σ for Rubin in several cases).

- [ ] **MUST RESOLVE:** §3.1 P5 currently says the pipeline's angular scales are
  "comfortably within the regime where Limber is accurate to well below the
  statistical precision of the survey." That may be wrong at Roman precision.
  Read this paper and correct or qualify the sentence. Vivian is an author, so
  this is the single most likely thing to get caught.
- [ ] **STRONG CITE** for §3.2 P4 (theory error must be a negligible fraction of
  statistical error) and §3.3 P2/P8 (nonlinear P(k) accuracy matters). This is
  near-perfect support for the thesis premise, from your own group.
- [ ] Check whether the P(k) part of that paper compares Takahashi vs HMcode
  directly. If it does, §3.3 P7 should cite it alongside your ratio figure.

### (b) Survey-landscape facts confirmed (2026)

- **DES-Y6 cosmic shear exists:** arXiv:2602.10065 (Feb 2026). Your §3.2 P2 claim
  that DES published its final-year shear cosmology is correct. There is also a
  DES-Y6 3x2pt paper (2601.14559) and an analysis-framework paper (2601.14859).
- **KiDS-Legacy:** Wright et al., arXiv:2503.19441, A&A 703, A158 (2025).
  S8 = 0.815 (+0.016/-0.021), agreeing with Planck at 0.73σ. This supports the
  "the most recent analyses have softened it" phrasing in §3.2 P2, and supports
  your original draft line about some claiming it ends the S8 tension.
- **HSC-Y3 real-space is Li et al. 2023, arXiv:2304.00702** (PRD 108, 123518),
  "Cosmology from Cosmic Shear Two-point Correlation Functions." This is the
  one to cite for a real-space analysis. Dalal et al. is the harmonic-space
  companion. CONFIRMED.
- **Eifler Roman forecasts:** 2004.05271 = "Cosmology with WFIRST/Roman:
  Multi-Probe Strategies" (MNRAS 507, 1746) is the HLS forecast paper, including
  the 2000 deg^2 reference survey. 2004.04702 is the separate Rubin-synergies
  paper (MNRAS 507, 1514). **Use 2004.05271** unless Evan says otherwise.
- **Roman ETC:** Hirata et al. 2012, arXiv:1204.5151, "The WFIRST Galaxy Survey
  Exposure Time Calculator." This is the citation for your n_eff figure.

---

## 1. Paste-ready Zotero list

Zotero: **Add Item(s) by Identifier** (the magic-wand icon). It accepts arXiv
IDs and DOIs, multiple at a time on separate lines. Paste this block, then
sort the new items into a per-chapter collection.

```
astro-ph/9912508
1411.0115
0809.5112
0705.0166
astro-ph/0406275
1407.6990
astro-ph/9904153
0910.3786
1809.01669
astro-ph/0609591
2105.13543
2105.13544
2105.13549
2602.10065
2007.15633
2503.19441
2304.00702
1503.03757
1204.5151
2004.05271
0805.2366
2606.23781
astro-ph/9911177
2010.11288
astro-ph/0206508
astro-ph/9611107
astro-ph/0207664
1208.2701
1505.07833
2009.01858
```

### What each one is, and where it is used

| Identifier | Paper | Used in |
|---|---|---|
| astro-ph/9912508 | Bartelmann & Schneider 2001, lensing review | §3.1 P2–P4 + appendix (anchor) |
| 1411.0115 | Kilbinger 2015, cosmic shear review | §3.1 P1 |
| 0809.5112 | LoVerde & Afshordi 2008, extended Limber | §3.1 P5 |
| 0705.0166 | Bridle & King 2007, NLA | §3.1 P8 |
| astro-ph/0406275 | Hirata & Seljak 2004, linear alignments | §3.1 P8 |
| 1407.6990 | Troxel & Ishak 2015, IA review (optional) | §3.1 P8 |
| astro-ph/9904153 | Hu 1999, shear tomography | §3.1 P7 |
| 0910.3786 | Krause & Hirata 2010, Born/higher-order | §3.1 P2 |
| 1809.01669 | Mandelbaum 2018, shape measurement review | §3.1 P1, P3 |
| astro-ph/0609591 | Albrecht et al. 2006, DETF report | §3.2 P1 |
| 2105.13543 | Amon et al. 2022, DES-Y3 shear | §3.2 P2 |
| 2105.13544 | Secco et al. 2022, DES-Y3 shear | §3.2 P2 (cite with above) |
| 2105.13549 | DES 2022 3x2pt | §3.2 P4 (Δχ² convention — see worklist) |
| 2602.10065 | DES-Y6 cosmic shear (2026) | §3.2 P2 |
| 2007.15633 | Asgari et al. 2021, KiDS-1000 | §3.2 P2 |
| 2503.19441 | Wright et al. 2025, KiDS-Legacy | §3.2 P2 |
| 2304.00702 | Li et al. 2023, HSC-Y3 real-space | §3.2 P2 |
| 1503.03757 | Spergel et al. 2015, WFIRST/Roman | §3.2 P3 |
| 1204.5151 | Hirata et al. 2012, Roman/WFIRST ETC | §3.2 P3 (n_eff) |
| 2004.05271 | Eifler et al. 2021, Roman multi-probe | §3.2 P3/P4 |
| 0805.2366 | Ivezić et al. 2019, LSST | §3.2 P3 |
| 2606.23781 | Zhang, …, Miranda et al., model approximations | §3.1 P5, §3.2 P4, §3.3 P2/P8 |
| astro-ph/9911177 | Lewis, Challinor & Lasenby 2000, CAMB | §3.3 P1, §4.1 |
| 2010.11288 | EuclidEmulator2 | §3.3 P3, §4.3 |
| astro-ph/0206508 | Cooray & Sheth 2002, halo model review | §3.3 P4 |
| astro-ph/9611107 | Navarro, Frenk & White 1997, NFW | §3.3 P4 |
| astro-ph/0207664 | Smith et al. 2003, Halofit | §3.3 P5 |
| 1208.2701 | Takahashi et al. 2012, "Taka" | §3.3 P5 |
| 1505.07833 | Mead et al. 2015, HMcode | §3.3 P6 |
| 2009.01858 | Mead et al. 2021, HMcode-2020, "Mead" | §3.3 P6 |

---

## 2. Items with no arXiv ID (add by DOI or by hand)

```
10.1098/rsta.1920.0009
10.1126/science.84.2188.506
10.1086/145672
```

| DOI | Paper | Used in |
|---|---|---|
| 10.1098/rsta.1920.0009 | Dyson, Eddington & Davidson 1920, the 1919 eclipse | §3.1 P1 |
| 10.1126/science.84.2188.506 | Einstein 1936, Science 84, 506, lens-like action | §3.1 P1 |
| 10.1086/145672 | Limber 1953, ApJ 117, 134 (optional) | §3.1 P5 |

Also, popular books cited in the §3.1 P1 footnote (add by ISBN if you keep them):
*No Shadow of a Doubt* (Kennefick) and *Proving Einstein Right* (Gilmore & Vaughan).

---

## 3. Still to resolve yourself

- [ ] **Birrer / TDCOSMO** — pick the specific paper you want for strong-lensing
  time-delay cosmography (§3.1 P1). Simon Birrer is on your committee, so choose
  deliberately.
- [ ] **OSU PSF / shape-measurement work** (§3.2 P3). Candidates found:
  arXiv:2607.09862 (the one you named), 2603.15763 (chromatic PSF effects on
  Roman shear), 2607.09849 (robust photometry for Roman HLIS). Pick one or two.
- [ ] **Troxel et al. 2021**, MNRAS 501, 2044, synthetic Roman HLIS simulation
  suite and wavefront errors — possible extra support for §3.2 P3.
- [ ] **arXiv:2601.00438**, Fisher forecasts for Roman HLIS 3x2pt — check whether
  this supersedes Eifler 2021 as your forecast reference.
- [ ] `hu_structure_1998` and `peacock_halofit_2014` — already in Zotero; place
  or cut (see worklist §2).

---

## 4. Citation workflow for the whole thesis

### Structure: ONE `references.bib`
Decided 2026-07-16. Use a single bib for the whole thesis, not per-chapter files.
Papers recur across chapters (CAMB in §3.3 and §4.1; Bartelmann & Schneider in
§3.1 and the appendix; EuclidEmulator2 in §3.3 and §4.3), and per-chapter bibs
produce duplicate entries under different keys. BibTeX only emits entries you
actually cite, so a single large bib costs nothing in the compiled document.

Migration: export your existing ch2 bib and the new Ch 3 entries into one
`references.bib`, then check for duplicate keys before deleting the old files.

### Organize in Zotero, not in the filesystem
Make Zotero **collections** per chapter (`Thesis/Ch2`, `Thesis/Ch3`, …). An item
can sit in several collections at once, which is exactly the recurrence problem
that per-chapter bibs handle badly. Export the parent `Thesis` collection to a
single `references.bib`.

### Use Better BibTeX
Install the Better BibTeX plugin, and:
- **Pin citation keys** so they never churn when you edit metadata. A key that
  silently changes breaks every `\cite` that used it.
- **Set up auto-export** of the `Thesis` collection to `references.bib`, so the
  file is always current without a manual export step.
- Pick one key format and keep it (e.g. `authorYearShorttitle`), matching
  whatever your ch2 keys already use so you don't have to rewrite those.

### The per-chapter harvest loop
This is the loop that produced this document; repeat it per chapter.
1. Draft the section with `% CITE-SUGGESTION:` markers inline wherever a citation
   belongs, rather than stopping to look things up mid-draft.
2. When the chapter is drafted, grep the `.tex` for `CITE-SUGGESTION` and
   `TODO-REF` and turn the hits into a lookup list like §1 above.
3. Resolve ambiguous ones (which of two companion papers? does the paper exist
   yet?) before adding, so you don't import the wrong item.
4. Add by identifier in Zotero, file into the chapter collection, export.
5. Compile and chase every undefined reference warning.
6. Check for orphans: entries you added but never cited. Orphan cites in a
   bibliography invite questions you don't want.

### Anticipated citation clusters for the remaining chapters
So you can pre-plan rather than discovering these late.
- **Ch 1 (write last):** mostly re-cites of Ch 2/3 anchors. Little new.
- **Ch 4 (Inference and Emulation):** Bayesian inference and samplers (Cobaya,
  MCMC/nested sampling, Gelman-Rubin); the CosmoLike/Cocoa pipeline (ask Vivian
  for the correct Cocoa citation, this is on your known-open-items list);
  neural network methods (ResMLP/residual networks, Adam, batch/layer norm,
  activation choices); the emulator landscape (CosmoPower, EuclidEmulator2,
  Mira-Titan, and any cosmic-shear DV emulators your group has published);
  transfer learning foundations (the canonical "how transferable are features
  in deep networks" line of work, plus fine-tuning/freezing literature).
- **Ch 5 (Experiments):** lightest citation load, mostly self-referential to your
  own results, plus the Δχ² convention and any TL-in-cosmology precedents.
- **Ch 6 (Conclusion):** near-zero new citations.

The heaviest remaining lift is Ch 4. Start collecting the ML references early,
because they are the ones least likely to already be in your Zotero.
