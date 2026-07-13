"""
Convert .1.txt dv generation output files to train_emulator-compatible format.
Strips weights, lnp (first 2 cols) and chi2* (last col), fixes header.

Usage: python clean_all_params.py --base_dir <chains_dir>
       python clean_all_params.py  # uses current directory as base
"""
import numpy as np
import os
import sys
import argparse

def clean_params(src, dst):
    print(f"Processing: {src}")
    with open(src) as f:
        lines = f.readlines()

    # line 0: '# nwalkers=45', line 1: '# weights lnp As_1e9 ... chi2*'
    header_cols = lines[1].strip().split()[1:]  # strip '#'
    keep = [i for i, c in enumerate(header_cols) if c not in ('weights', 'lnp', 'chi2*')]
    col_names = [header_cols[i] for i in keep]

    data = np.loadtxt(src)[:, keep]
    np.savetxt(dst, data, header=' '.join(col_names), comments='# ', fmt='%.18e')
    print(f"  Done: {data.shape} -> {dst}")
    return col_names

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', default='.', help='Base chains directory')
args = parser.parse_args()
base = args.base_dir

files = [
    # (subdir, src_filename, dst_filename)
    ('taka_T200', 'roman_real_lcdm_taka_train_T200_cs_200.1.txt',  'roman_real_lcdm_b_taka_train_parameters_T200.txt'),
    ('taka_T200', 'roman_real_lcdm_taka_valid_T100_cs_100.1.txt',  'roman_real_lcdm_b_taka_valid_parameters_T100.txt'),
    ('taka_T200', 'roman_real_lcdm_taka_test_T100_cs_100.1.txt',   'roman_real_lcdm_b_taka_test_parameters_T100.txt'),
    ('taka_T500', 'roman_real_lcdm_taka_train_T500_cs_500.1.txt',  'roman_real_lcdm_b_taka_train_parameters_T500.txt'),
    ('taka_T500', 'roman_real_lcdm_taka_valid_T250_cs_250.1.txt',  'roman_real_lcdm_b_taka_valid_parameters_T250.txt'),
    ('taka_T500', 'roman_real_lcdm_taka_test_T250_cs_250.1.txt',   'roman_real_lcdm_b_taka_test_parameters_T250.txt'),
    ('mead_T500', 'roman_real_lcdm_mead_train_T500_cs_500.1.txt',  'roman_real_lcdm_b_mead_train_parameters_T500.txt'),
    ('mead_T500', 'roman_real_lcdm_mead_valid_T250_cs_250.1.txt',  'roman_real_lcdm_b_mead_valid_parameters_T250.txt'),
    ('mead_T500', 'roman_real_lcdm_mead_test_T250_cs_250.1.txt',   'roman_real_lcdm_b_mead_test_parameters_T250.txt'),
]

for subdir, src_name, dst_name in files:
    src = os.path.join(base, subdir, src_name)
    dst = os.path.join(base, subdir, dst_name)
    if not os.path.exists(src):
        print(f"SKIP (not found): {src}")
        continue
    if os.path.exists(dst):
        print(f"SKIP (already exists): {dst}")
        continue
    clean_params(src, dst)

print("\nAll done.")
