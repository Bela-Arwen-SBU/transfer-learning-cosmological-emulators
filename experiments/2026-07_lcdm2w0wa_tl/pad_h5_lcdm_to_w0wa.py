#!/usr/bin/env python3
"""Build the padded 17-input companion .h5 for the LCDM->w0wa TL base.

Why: in TL mode train_emulator.py loads input normalization (sample_mean,
sample_std) from the pretrained .h5. The LCDM .h5 arrays are 15-wide; the
w0wa model input is 17-wide (ord: ...A1_2, w, w0pwa) -> shape mismatch crash.

What this does:
  1. copies every dataset from the source LCDM .h5 unchanged,
  2. except sample_mean / sample_std, which get two entries APPENDED
     in ord order [w, w0pwa]:
        mean  -> plain mean of each column from the w0wa train params txt
        std   -> 5.0 * sigma, matching the 5x bake-in the pipeline applies
                 when it saves sample_std (train_emulator save step)

In the params txt header the columns are (0)As_1e9 ... (4)omegam
(5)w0pwa (6)w (7..)DZ/A1 — note txt order is w0pwa,w but model ord order
is w,w0pwa; this script appends in MODEL ord order [w, w0pwa].

Usage:
    python pad_h5_lcdm_to_w0wa.py <source_lcdm.h5> <w0wa_train_params.txt> <output_padded17.h5>

Read-only on inputs; refuses to overwrite an existing output.
Reading the 2.1 GB train txt takes a few minutes.
"""
import os
import sys

import h5py
import numpy as np

TXT_COL_W0PWA = 5   # from the verified header: # As ns H0 omegab omegam w0pwa w ...
TXT_COL_W = 6
EXPECTED_IN = 15
STD_BAKE_IN = 5.0   # pipeline stores sample_std = 5.0 * sigma


def main(src_h5, train_txt, dst_h5):
    if os.path.exists(dst_h5):
        sys.exit(f"refusing to overwrite existing {dst_h5}")
    for p in (src_h5, train_txt):
        if not os.path.exists(p):
            sys.exit(f"input not found: {p}")

    print(f"reading w0pwa,w columns from {train_txt} (a few minutes for the 2.1 GB train file)...")
    cols = np.loadtxt(train_txt, usecols=(TXT_COL_W0PWA, TXT_COL_W))
    w0pwa, w = cols[:, 0], cols[:, 1]
    print(f"  {len(w)} rows")
    print(f"  w     : mean {w.mean():.6f}  sigma {w.std():.6f}")
    print(f"  w0pwa : mean {w0pwa.mean():.6f}  sigma {w0pwa.std():.6f}")

    # appended in MODEL ord order: [w, w0pwa]
    add_mean = np.array([w.mean(), w0pwa.mean()])
    add_std = STD_BAKE_IN * np.array([w.std(), w0pwa.std()])

    with h5py.File(src_h5, "r") as fin, h5py.File(dst_h5, "w") as fout:
        print(f"\nsource {src_h5} datasets:")
        for key in fin:
            data = fin[key][()]
            shape = np.shape(data)
            print(f"  {key}: {shape}")
            if key in ("sample_mean", "sample_std"):
                arr = np.asarray(data, dtype=np.float64)
                if arr.shape[-1] != EXPECTED_IN:
                    sys.exit(f"{key} last dim is {arr.shape[-1]}, expected {EXPECTED_IN} — aborting")
                extra = add_mean if key == "sample_mean" else add_std
                extra = extra.reshape(arr.shape[:-1] + (2,)) if arr.ndim > 1 else extra
                new = np.concatenate([arr, extra], axis=-1)
                fout[key] = new
                print(f"    -> padded to {new.shape}; appended [w, w0pwa] = {np.asarray(extra).ravel()}")
            else:
                fout[key] = data

    with h5py.File(dst_h5, "r") as f:
        assert f["sample_mean"][()].shape[-1] == 17
        assert f["sample_std"][()].shape[-1] == 17
    print(f"\nwrote {dst_h5}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
