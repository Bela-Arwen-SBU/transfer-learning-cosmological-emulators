#!/usr/bin/env python3
"""Patch train_emulator.py to support column-masked input training.

Adds a MASK_INPUT_COLS environment variable: when set to N in transfer
learning mode, a gradient hook zeros the gradient on the first N input
columns of model.0.weight, so the pretrained embedding stays fixed and only
the new input columns train. Off by default (env var unset); existing
behavior is untouched. Requires weight_decay=0 (the pipeline default),
otherwise Adam's decay bypasses the hook.

Inserts after the freeze_layers() call, before the optimizer is built.
Idempotent; backs up to train_emulator.py.bak; syntax-checks before writing.

Usage: python3 apply_mask_patch.py /path/to/train_emulator.py
Applied to NvWulf 2026-07-12.
"""
import ast
import re
import shutil
import sys

p = sys.argv[1] if len(sys.argv) > 1 else \
    "/lustre/nvwulf/projects/MirandaGroup-nvwulf/bela/cocoa/Cocoa/train_emulator.py"
src = open(p).read()
if "MASK_INPUT_COLS" in src:
    raise SystemExit("already patched - nothing to do")
lines = src.splitlines(keepends=True)
hits = [i for i, l in enumerate(lines) if "freeze_layers(" in l and "frozen_params" in l]
assert len(hits) == 1, f"expected exactly 1 anchor line, found {len(hits)} - aborting"
idx = hits[0]
indent = re.match(r"\s*", lines[idx]).group(0)
block = "".join(indent + l + "\n" for l in [
    "# Column-masked input training (set MASK_INPUT_COLS=15 to lock the",
    "# pretrained input columns and train only the new ones)",
    "mask_input_cols = int(os.environ.get('MASK_INPUT_COLS', '0'))",
    "if transfer_learning and mask_input_cols > 0:",
    "    w0 = model.model[0].weight",
    "    w0.requires_grad = True",
    "    col_mask = torch.zeros_like(w0)",
    "    col_mask[:, mask_input_cols:] = 1.0",
    "    w0.register_hook(lambda g, m=col_mask: g * m.to(g.device))",
    "    model.model[0].bias.requires_grad = False",
    "    print(f'TRANSFER LEARNING: input columns 0-{mask_input_cols-1} locked; '",
    "          f\"training only the last {w0.shape[1] - mask_input_cols} input columns\")",
])
lines.insert(idx + 1, block)
out = "".join(lines)
if not re.search(r"^import os\b", out, re.M):
    first = next(i for i, l in enumerate(lines) if re.match(r"^(import|from) ", l))
    lines.insert(first, "import os\n")
    out = "".join(lines)
ast.parse(out)
shutil.copy2(p, p + ".bak")
open(p, "w").write(out)
print(f"patched at line {idx + 2}; backup: {p}.bak")
