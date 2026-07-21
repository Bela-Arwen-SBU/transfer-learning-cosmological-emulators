# Chapter 3 worklist — a day's work to lock the chapter down

Generated 2026-07-16 as a companion to the drafted §3.1 / §3.2 / §3.3.
Everything here is something concrete you can do without me. Check boxes as you go.
Order is roughly "highest leverage first," but the reference hunt (§2) and the
paper-number verifications (§3) are the two big time sinks, so budget for those.

Legend: `[HAVE]` = bibkey already in Zotero (per scaffold); `[ADD]` = needs adding;
`[VERIFY]` = confirm the fact/number/identity before it ships to Vivian.

---

## 1. Confirmed pipeline facts (reference card — these are DONE, no action)

Pulled and cross-checked from the repo configs on 2026-07-16. Use these anywhere
in Ch 3 / Ch 5 and they will be consistent.

| Quantity | Value | Source file |
|---|---|---|
| Source tomographic bins `N_tomo` | 8 | `ones.dataset`, `roman_DZ_S1..S8` |
| Angular bins `N_theta` | 15 | `ones.dataset` |
| Theta range | 2.5' – 250', log-spaced | `ones.dataset` |
| Cosmic-shear vector length | 1080 = (8·9/2)·2·15 = 36·2·15 | README_w0wa_taka_run1 (`indices 0:1080`) |
| Full 3×2pt vector | 2115 (1080 shear + 915 GGL + 120 clustering) | README + ggl_exclude |
| Emulator input dim | 15 (ΛCDM) / 17 (w0wa) | `.paramnames`, w0wa README |
| — ΛCDM inputs | 5 cosmo + 8 photo-z (DZ) + 2 IA | `roman_emulator_cs_taka.yaml` `ord` |
| — w0wa adds | `w0pwa`, `w` | w0wa README, 17 total |
| IA model | NLA, 2 sampled: `A1_1` (amp) + `A1_2` (z-power); `A2`,`BTA` fixed 0 | roman YAMLs |
| Shear calib `m^i` | 8, fixed to 0 in training, applied analytically (fast_params) | `fast_params` |
| Taka string | CAMB `halofit_version: takahashi` | `lcdm_takahashi_cs_CNN.yaml` |
| Mead string | CAMB `halofit_version: mead2020` (NOT `mead2020_feedback`) | `roman_emulator_cs_mead.yaml` |
| Mead baryons | OFF — gravity-only / DMONLY, no `logT_AGN` set anywhere | all mead configs |

Note (harmless): `roman_emulator_cs_taka.yaml` and `..._taka_resmlp_N10000.yaml`
have `halofit_version: mead2020` in their `theory:` block despite the "taka"
filename. That block is the post-training theory stub, not the dataset generator,
so it does not affect the Taka datasets. Canonical generators are the `*_CNN.yaml`
and `*_nvwulf.yaml` files.

---

## 2. References to add (the big task)

Go through these on ADS, add to `references.bib`, and make the bibkeys match what
Zotero exports. Each row is a `% CITE-SUGGESTION:` I left in the draft. arXiv IDs
are starting points — confirm each on ADS, some I flagged as uncertain.

### §3.1 — Weak lensing
- [ ] `[ADD]` Dyson, Eddington & Davidson 1920 — the 1919 eclipse measurement (P1 history)
- [ ] `[ADD]` Einstein 1936, Science 84, 506 — lens-like action of a star (P1)
- [ ] `[ADD/HAVE?]` Birrer et al. (TDCOSMO) — strong-lensing time-delay cosmography (P1). Pick the specific TDCOSMO paper you want.
- [ ] `[ADD]` Mandelbaum 2018, ARA&A 56, 393 (arXiv:1809.01669) — shape measurement / shape noise (P1, P3)
- [ ] `[ADD]` Bartelmann & Schneider 2001, Phys.Rep. 340, 291 (astro-ph/9912508) — **formalism anchor**, used P2–P4 + appendix
- [ ] `[ADD]` Kilbinger 2015, Rep.Prog.Phys. (arXiv:1411.0115) — cosmic shear survey review (P1)
- [ ] `[ADD]` Krause & Hirata 2010 (arXiv:0910.3786) — Born / higher-order lensing corrections (P2)
- [ ] `[ADD]` LoVerde & Afshordi 2008 (arXiv:0809.5112) — extended Limber accuracy (P5)
- [ ] `[ADD, optional]` Limber 1953 — original Limber approximation (P5, decide if you want the historical cite)
- [ ] `[ADD]` Hu 1999 (astro-ph/9904153) — shear tomography as the dark-energy handle (P7)
- [ ] `[ADD]` Hirata & Seljak 2004 (astro-ph/0406275) — linear-alignment IA (P8)
- [ ] `[ADD]` Bridle & King 2007 (arXiv:0705.0166) — NLA model (P8)
- [ ] `[ADD, optional]` Troxel & Ishak 2015 (arXiv:1407.6990) — IA review (P8)

### §3.2 — Surveys
- [ ] `[ADD]` Albrecht et al. 2006 (astro-ph/0609591) — **DETF report**, Stage I–IV + figure of merit (P1)
- [ ] `[ADD]` Amon et al. 2022 (arXiv:2105.13543) + Secco et al. 2022 (arXiv:2105.13544) — DES-Y3 shear pair, cite together (P2)
- [ ] `[VERIFY]` DES-Y6 shear cosmology — does the final-year shear paper exist yet? If not, soften "published its final-year shear cosmology" in P2.
- [ ] `[ADD]` Asgari et al. 2021 (arXiv:2007.15633) — KiDS-1000 (P2)
- [ ] `[VERIFY]` KiDS-Legacy / DR5 reference — Wright et al. 2024/2025? Confirm exact citation (P2)
- [ ] `[VERIFY]` HSC-Y3: **Li et al. 2023 (real-space)** vs Dalal et al. 2023 (harmonic). Cite Li to match your real-space analysis; confirm which arXiv ID is which (P2)
- [ ] `[ADD, optional]` Abdalla et al. 2022 Snowmass tensions (arXiv:2203.06142) — one S8-tension summary cite (P2)
- [ ] `[ADD]` Spergel et al. 2015 (arXiv:1503.03757) — WFIRST/Roman mission + reference survey (P3)
- [ ] `[ADD]` Roman Exposure Time Calculator (Hirata et al.) — source of the n_eff figure (P3). Find the citable reference.
- [ ] `[ADD]` Ivezić et al. 2019 (arXiv:0805.2366) — LSST (P3)
- [ ] `[ADD, optional]` OSU PSF / shape-measurement work — Effortless/FastImCOM (arXiv:2607.09862) and Caili's related papers (P3)
- [ ] `[VERIFY]` Eifler et al. 2021 Roman forecasts — 2004.04702 vs 2004.05271. Pick the one matching the group's forecast lineage (ask Evan). (P4 context)
- [ ] `[VERIFY]` DES Collaboration 2022 3×2pt (arXiv:2105.13549) — source of the Δχ² < 0.2 convention. See decision in §4 below. (P4)
- [ ] `[HAVE]` DESI DR2 — reuse the same key already used in §2.4 (P5)

### §3.3 — Nonlinear power spectrum
- [ ] `[ADD]` Lewis, Challinor & Lasenby 2000 (astro-ph/9911177) — CAMB (P1; reuse in §4.1)
- [ ] `[ADD]` EuclidEmulator2 (arXiv:2010.11288) — example P(k) emulator (P3; also §4.3, keep ≤2 uses)
- [ ] `[ADD]` Cooray & Sheth 2002 (astro-ph/0206508) — halo-model review (P4)
- [ ] `[ADD]` Navarro, Frenk & White 1997 (astro-ph/9611107) — NFW profile (P4)
- [ ] `[ADD]` Smith et al. 2003 (astro-ph/0207664) — original Halofit (P5)
- [ ] `[ADD]` Takahashi et al. 2012 (arXiv:1208.2701) — Halofit recalibration = "Taka" (P5)
- [ ] `[ADD]` Mead et al. 2015 (arXiv:1505.07833) — HMcode (P6)
- [ ] `[ADD]` Mead et al. 2021 (arXiv:2009.01858) — HMcode-2020 = "Mead" (P6)
- [ ] `[HAVE, optional]` peacock_halofit_2014 — supporting halofit cite (P5)

### Orphan [HAVE] keys from the scaffold — decide keep or cut
- [ ] `hu_structure_1998` — if it's Hu (1998) on structure/lensing power spectra, best fit is §3.1 P5 (Limber/C(ell)). Place it or cut it; don't leave dangling.
- [ ] `lesgourgues_cosmic_2011` (CLASS) — you use CAMB, not CLASS. Drop unless you mention CLASS elsewhere (orphan cites invite questions).
- [ ] `prat_dark_2025`, `doux_going_2025` — check abstracts; if survey/forecast papers they slot into §3.2, else cut from Ch 3.
- [ ] `schneider_baryonification_2025` — only cite in §3.3 P6 if you say something about baryons; since Mead runs gravity-only, you probably don't. Likely cut from Ch 3.

---

## 3. Fact / number verifications (do NOT ship these from memory)

- [ ] `[VERIFY]` **Takahashi 2012 accuracy** — bracketed as "~5% for k ≲ 1 h/Mpc" in §3.3 P5. Pull the exact figure, k-range, and z-range from the paper.
- [ ] `[VERIFY]` **HMcode-2020 accuracy** — bracketed as "~2.5% for k ≲ 10 h/Mpc" in §3.3 P6. Confirm from Mead et al. 2021.
- [ ] `[VERIFY]` **Roman launch window** — §3.2 P3 has a placeholder; your draft said Aug 2026. Use NASA's current "no later than" date.
- [ ] `[VERIFY]` **Roman survey area** — §3.2 P3 placeholder (~2000 deg²?). The covariance-generation config is NOT in this repo; get the number the pipeline actually assumes, or cite Spergel's reference survey and say so.
- [ ] `[VERIFY]` **n_eff = ~17 gal/arcmin²** — §3.2 P3. Your ETC note had n=18.32, n_eff=16.81 gal/arcmin². Confirm which one the pipeline covariance uses and state that one.
- [ ] `[VERIFY]` **The worked-k footnote numbers** (§3.1 P4): χ~1.7 Gpc/h at z~1, ℓ~10³ → k~0.6 h/Mpc, θ~10'. These are round illustrative numbers; recompute with the pipeline cosmology if you keep the footnote (or cut it — see decisions).

---

## 4. Open decisions (yours to make)

- [ ] **Δχ² < 0.2 provenance** (§3.2 P4) — is the 0.2 really from DES 3×2pt (2105.13549), or is it group-internal folklore traceable to Evan's Part I? Confirm the exact number and the dof it refers to. Committee may probe this. **Ask Vivian / Evan.**
- [ ] **Appendix format** — the geodesic derivation is written as `\chapter{}` assuming an `\appendix` block. If your appendices are `\section`s, tell me and I'll convert. Confirm your template's convention.
- [ ] **IA notation** — draft §3.1 P8 uses `A_IA, eta_IA`; config uses `roman_A1_1`, `roman_A1_2`. Decide the thesis-wide symbol and align with the Ch 5 input table.
- [ ] **Keep the worked-k footnote?** (§3.1 P4) — you liked "add it and take it out." Decide now if it stays for Teaney or goes.
- [ ] **§3.2 length** — currently ~2.5 pp and deliberately tight. Decide if you want P2 (per-survey detail) or P3 (space-vs-ground) expanded, or leave lean.
- [ ] **3×2pt framing** — you chose "mention the extraction" (done in §3.1 P7). Confirm you're happy with 2115→1080 stated in Ch 3 vs deferring to Ch 5.

---

## 5. Figure task (SeaWulf — not a chat task)

- [ ] **`fig:pk_ratio`** (§3.3 P7): ratio P_mead2020 / P_takahashi vs k (~0.01–10 h/Mpc) at z = 0, 0.5, 1 (adjust z to what the lensing kernel weights), at the fiducial cosmology from `roman_emulator_cs`, both prescriptions gravity-only, same CAMB settings as the pipeline. Save to `figures/pk_ratio_mead_taka.pdf`.
  - When ready, I can write the generation script against the pipeline's exact CAMB settings — flag me for it.

---

## 6. Consistency / cross-reference checks

- [ ] Confirm every `\ref`/`\label` in Ch 3 resolves: `ch:cosmology`, `ch:inference`, `ch:experiments`, `sec:wl`, `sec:surveys`, `sec:nonlinear`, `sec:desi`, `sec:likelihood`, `app:geodesic`, `fig:pk_ratio`, plus all `eq:` labels.
- [ ] `sec:likelihood` (referenced from §3.2 `eq:delta_chi2` and §3.3) must be the actual §4.1 label. Fix if named differently.
- [ ] `eq:delta_chi2` is defined in §3.2 and should be `\eqref`'d (not redefined) in §4.1.
- [ ] Data-vector numbers (1080, N_tomo=8, N_theta=15) must match the Ch 5 methods table verbatim. Reconcile once the Ch 5 table exists.
- [ ] Input-count story (15 ΛCDM / 17 w0wa) must match the Ch 5 input table and the §2.4 dark-energy parameter discussion.
- [ ] Back-references into Ch 2 (the `H^2(a)` display, the δ definition, A_s/n_s, S_8 foregrounding) point at real equation labels once Ch 2 is finalized.

---

## 7. Prose read-through prompts (read critically, in your own voice)

- [ ] §3.1 P1 — is the Einstein/Eddington history accurate and the way you want it told? (1911 half-value, 1915 full, 1919 confirm, 1936 lens paper.) Trim if it drags.
- [ ] Do an em-dash / AI-phrasing pass on the parts YOU wrote earlier in the scaffold (the drafted prose from me should already be clean; your older sentences may not be).
- [ ] §3.3 P7 is the load-bearing paragraph (why TL should work + why it's hard). Make sure it says what you mean before Ch 5 leans on it.
- [ ] Check that the three "seed → payoff" links read clearly: broad kernel (§3.1 P4) → k_max diagnostic; J_4 asymmetry (§3.1 P6) → hardest-to-emulate ξ_-; parameter-dependent reshaping (§3.3 P7) → correction-network failure.
- [ ] §3.2: decide if the S8-tension sentence stays one sentence (scaffold says don't elaborate).

---

## Suggested order for the day
1. References (§2) — longest, do it first while fresh. ~2–3 hrs for ~30 keys.
2. Paper-number verifications (§3) — read Takahashi 2012 + Mead 2021 for the accuracy figures while you're in ADS.
3. Decisions (§4) — quick once you've seen the sources; email Vivian/Evan re Δχ² provenance early so you're not blocked.
4. Consistency pass (§6) — compile and chase every undefined `\ref`.
5. Prose read-through (§7) — last, with fresh eyes.
6. Figure (§5) — separate SeaWulf session; flag me to write the script.
