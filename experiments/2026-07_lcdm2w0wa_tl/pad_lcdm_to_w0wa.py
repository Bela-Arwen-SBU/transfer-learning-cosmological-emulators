#!/usr/bin/env python3
"""Pad a 15-input LCDM emulator checkpoint to 17 inputs for w0wa TL.

Model input ordering comes from the yaml 'ord' list, not the data txt header
(train_emulator reorders data columns by name to match 'ord'):

LCDM ord (15): As_1e9 ns H0 omegab omegam DZ_S1..S8 A1_1 A1_2
w0wa ord (17): the same 15 in the same order, then w, w0pwa appended

So we append two zero columns at positions 15,16 of model.0.weight. The
padded model initially ignores (w, w0pwa) and reproduces its LCDM prediction
exactly. All other tensors are unchanged. Writes <out>.pt; the companion .h5
(needed by the sweep preflight) is copied from the source model.

Usage:
    python pad_lcdm_to_w0wa.py <input.pt> <output.pt>

Read-only on the input; refuses to overwrite an existing output.
"""
import os
import shutil
import sys

import torch

INSERT_AT = 15  # w, w0pwa are appended at the END of the w0wa yaml 'ord' list
N_NEW = 2
EXPECTED_IN = 15


def main(src, dst):
    if os.path.exists(dst):
        sys.exit(f"refusing to overwrite existing {dst}")

    state = torch.load(src, map_location="cpu")
    if not isinstance(state, dict):
        sys.exit(f"unexpected checkpoint type: {type(state)}")

    key = "model.0.weight"
    if key not in state:
        sys.exit(f"{key} not found; keys: {list(state.keys())[:6]} ...")

    w = state[key]
    print(f"{key}: {tuple(w.shape)}")
    if w.shape[1] != EXPECTED_IN:
        sys.exit(f"expected input dim {EXPECTED_IN}, got {w.shape[1]} — aborting")

    zeros = torch.zeros(w.shape[0], N_NEW, dtype=w.dtype)
    state[key] = torch.cat([w[:, :INSERT_AT], zeros, w[:, INSERT_AT:]], dim=1)
    print(f"padded  : {tuple(state[key].shape)} (zero cols at {INSERT_AT},{INSERT_AT+1})")

    # sanity: identical values in the surviving columns
    assert torch.equal(state[key][:, :INSERT_AT], w[:, :INSERT_AT])
    assert torch.equal(state[key][:, INSERT_AT + N_NEW:], w[:, INSERT_AT:])
    assert torch.all(state[key][:, INSERT_AT:INSERT_AT + N_NEW] == 0)

    torch.save(state, dst)
    print(f"wrote {dst}")

    src_h5, dst_h5 = src[:-3] + ".h5", dst[:-3] + ".h5"
    if os.path.exists(src_h5) and not os.path.exists(dst_h5):
        shutil.copy2(src_h5, dst_h5)
        print(f"copied companion {dst_h5}")
    else:
        print(f"NOTE: companion .h5 not copied ({src_h5} exists: {os.path.exists(src_h5)})")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
