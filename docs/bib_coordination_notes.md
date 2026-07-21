# Bibliography coordination notes — for the Ch 1/2/3/5/6 chats

Written 2026-07-18 by the Ch 4 chat. Paste the relevant section into
each chat. The "rules for everyone" block applies to all of them.

---

## RULES FOR EVERYONE (read this first)

**1. The bibliography is now one file per chapter.**
`chapter1.bib` … `chapter6.bib` all exist. `main.tex` loads all six:

```latex
\bibliography{chapter1,chapter2,chapter3,chapter4,chapter5,chapter6}
```

**Nobody needs to edit `main.tex` again for bibliography reasons.** It is
on Béla's do-not-touch list; this was a one-time authorized change.

**2. `chapter2.bib` is the legacy thesis-wide file.** It holds 50 entries,
all Ch 2 dark-energy theory. Do NOT rename it — that breaks everyone.
From now on it is simply Ch 2's file.

**3. Key style — match the existing convention exactly.**
Zotero's default: `lastname_firsttitleword_year`, lowercase,
underscore-separated, leading articles dropped. Hyphens are fine
(`dinda_model-agnostic_2025`). Examples already in use:
`linder_mirage_2007`, `brout_pantheon_2022`, `riess_observational_1998`.

**4. Keys must be unique across all six files.** BibTeX reads them as one
database. Two entries with the same key is an error; two different keys
for the same paper is worse, because it silently produces a duplicated
bibliography entry.

**5. Ownership rule: the chapter that cites a work FIRST owns the entry.**
Every later chapter reuses that key and adds nothing. Before adding any
entry, run:

```bash
cd ~/thesis-overleaf-live && grep -l "author.*Surname" chapter*.bib
```

**6. Check your work.** After editing, compile and read the BibTeX log:

```bash
cd ~/thesis-overleaf-live && tectonic --keep-logs main.tex >/dev/null 2>&1
grep -i "didn't find a database entry" main.blg | sort -u
```

Every line is a `\cite` with no entry. The list should shrink to empty.

**7. Concurrent-work warning.** Several chats are editing this repo at
once. Commit and push your own files only (`git add <specific files>`,
never `git add -A`). Pull with `--rebase` before pushing. If a rebase is
blocked by someone else's unstaged file, wait rather than stashing it.

---

## ALREADY DONE — `chapter4.bib` (do not duplicate these)

15 verified entries. Six came from the INSPIRE-HEP API (DOI, journal,
volume, pages all authoritative); the rest are canonical stats/ML
references written by hand.

| Key | Work |
|---|---|
| `krause_cosmolike_2017` | CosmoLike |
| `torrado_cobaya_2021` | Cobaya |
| `metropolis_equation_1953` | Metropolis algorithm |
| `hastings_monte_1970` | Hastings generalization |
| `gelman_inference_1992` | Gelman–Rubin diagnostic |
| `zhong_attention-based_2025` | Emulator series I |
| `saraivanov_attention-based_2025` | Emulator series II (Evan) |
| `zhu_attention-based_2026` | Emulator series III (Yijie) |
| `hornik_multilayer_1989` | Universal approximation |
| `cybenko_approximation_1989` | Universal approximation (alternative) |
| `kingma_adam_2015` | Adam |
| `he_deep_2016` | ResNet |
| `pan_survey_2010` | Transfer-learning survey |
| `yosinski_how_2014` | Early-general / late-specific |
| `krishnaraj_transfer_2026` | TL between physical models |

---

## CH 1 — four placeholders are now resolvable, zero new entries needed

`ch1.tex` cites four TODO placeholder keys. All four are real entries in
`chapter4.bib` already. Swap the keys and you are done:

| Replace | With |
|---|---|
| `TODO-saraivanov-2024` | `saraivanov_attention-based_2025` |
| `TODO-zhu-2025` | `zhu_attention-based_2026` |
| `TODO-yosinski-2014` | `yosinski_how_2014` |
| `TODO-transfer-survey` | `pan_survey_2010` |

Note the year corrections: Saraivanov II is **2025** (PRD 111, 123520),
not 2024; Zhu III is **2026** (PRD 113, 103536), not 2025. Both were
published after the arXiv postings the old keys were named for.

Locations: `ch1.tex:36` (the two emulator papers) and `ch1.tex:52` (the
two transfer-learning references).

---

## CH 2 — three missing entries, one shared with Ch 3

BibTeX currently reports these as having no database entry:

- **`linder_exploring_2003`** — the long-standing undefined citation
  (appears on p.12 of the compiled PDF). This is the only broken
  reference that has survived every compile so far.
- **`perlmutter_measurements_1999`** — the Supernova Cosmology Project
  measurement paper.
- **`TODO-desi-dr2-keyproject`** — placeholder, used at `ch2.tex:220`.

⚠ **`TODO-desi-dr2-keyproject` is also used in `ch3.tex:256`.** Ch 2 owns
it. Pick the real key, put the entry in `chapter2.bib`, and tell the
Ch 3 chat what you chose so they update their `\citep` rather than
creating a second DESI entry. There is already a note in `ch3.tex`
flagging that the two must be fixed together.

---

## CH 3 — populate `chapter3.bib`, and you own CAMB

`chapter3.bib` exists as an empty stub (created by the Ch 4 chat so that
`main.tex` would not reference a missing file). It is yours to fill.

Keys already cited in `ch3.tex` with no entry yet:
`dyson_determination_1920`, `einstein_lens_1936`, `mandelbaum_weak_2018`,
`bartelmann_weak_2001`, `kilbinger_cosmology_2015` — plus whatever else
you add as you go. Run the `main.blg` check above for the live list.

**⚠ You own the CAMB citation.** Lewis, Challinor & Lasenby 2000,
`astro-ph/9911177`. Ch 3 §3.3 P1 introduces CAMB first, and its own
scaffold comment says *"Reuse the same key in Sec 4.1."* Ch 4 §4.2 names
CAMB and will cite your key. **Please put it in `chapter3.bib` and tell
the Ch 4 chat what the key is** (suggestion: `lewis_efficient_2000`).
Ch 4 has deliberately NOT added CAMB, to avoid a duplicate.

**Δχ² < 0.2 provenance is resolved** — there is an updated comment at
`ch3.tex:249`. Summary: the *value* 0.2 is the DES-Y3 threshold, so cite
DES Collaboration 2022 (2105.13549) for the number. The *group's*
criterion is the stronger compound claim ">90% of test points below
0.2", established across the emulator series — cite
`zhong_attention-based_2025`, `saraivanov_attention-based_2025`,
`zhu_attention-based_2026`, all already in `chapter4.bib`. Still
unverified: the exact dof the 0.2 refers to.

---

## CH 5 — you probably need zero new entries

Ch 5's likely citations are all already in `chapter4.bib`. Reuse these
keys; do **not** copy them into `chapter5.bib`:

`saraivanov_attention-based_2025` · `zhu_attention-based_2026` ·
`zhong_attention-based_2025` · `pan_survey_2010` · `yosinski_how_2014` ·
`krishnaraj_transfer_2026` · `he_deep_2016` · `kingma_adam_2015` ·
`torrado_cobaya_2021` · `krause_cosmolike_2017`

The one thing you may need to add: **DES Collaboration 2022**
(`2105.13549`) if you quote the Δχ² < 0.2 threshold directly rather than
referring back to Ch 3. Coordinate with the Ch 3 chat — whoever cites it
first owns it.

Ch 4 §4.4 has been written to supply your vocabulary: source/target,
base model, scratch vs fine-tuning vs feature extraction, the
`none`/`early_k`/`late_k`/`resnet_k` naming, column masking, the
embedding rescale, and correction networks (blind vs θ-conditioned).
Ch 4 states no Ch 5 numbers anywhere, deliberately.

---

## CH 6 — reuse only

The conclusion should re-cite works already introduced. Before adding
anything to `chapter6.bib`, grep the other five files. Realistically
this file may stay empty.

---

## ONE ITEM ONLY BÉLA CAN SUPPLY

**`lloyd_transfer_2026`** — the Ch 4 scaffold marks this `[HAVE]`,
meaning it should exist in Zotero. It is not in `chapter2.bib` and
returns **no match on INSPIRE-HEP**. It could not be verified or
reconstructed, and was deliberately not invented. Export it from Zotero
and paste it into `chapter4.bib`, or supply the title and authors.
