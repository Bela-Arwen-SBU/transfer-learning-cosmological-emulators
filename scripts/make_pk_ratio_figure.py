#!/usr/bin/env python3
"""
Figure fig:pk_ratio for thesis Sec 3.3 P7.

Plots the ratio of the nonlinear matter power spectrum from HMcode-2020
("Mead") to Halofit-Takahashi ("Taka") at the pipeline's fiducial cosmology,
for several redshifts. Both prescriptions are run GRAVITY-ONLY (no baryon
feedback), matching the emulator training pipeline.

RUN ON SeaWulf (needs CAMB + the same environment used to generate the
training data). Cosmology and CAMB settings are taken verbatim from
experiments/2026-02_roman_baseline/roman_emulator_cs_taka.yaml so the figure
is consistent with the emulator's inputs.

    module load anaconda/3      # or whatever provides your CAMB env
    conda activate <your-cocoa-env>
    python scripts/make_pk_ratio_figure.py

Output: figures/pk_ratio_mead_taka.pdf  (copy into the thesis figures/ dir)

If a number here disagrees with the config, TRUST THE CONFIG and update the
FIDUCIAL / CAMB_ARGS blocks below; they are transcribed, not authoritative.
"""

import numpy as np
import matplotlib.pyplot as plt
import camb

# ---------------------------------------------------------------------------
# Pipeline fiducial cosmology  (roman_emulator_cs_taka.yaml -> train_args.fiducial
# and the fixed params mnu, tau). omegab/omegam are DENSITY FRACTIONS; they are
# converted to physical densities below with the SAME formulas the config's
# derived-parameter block uses, so CAMB sees exactly what the pipeline feeds it.
# ---------------------------------------------------------------------------
FIDUCIAL = dict(
    As_1e9=2.10057526,     # 1e9 * A_s
    ns=0.959882119,
    H0=66.9443126,
    omegab=0.0500811280,   # Omega_b
    omegam=0.330931706,    # Omega_m (total matter incl. cdm + b + massive nu)
    mnu=0.06,              # sum of neutrino masses [eV], fixed
    tau=0.06,              # fixed
    w0=-1.0, wa=0.0,       # LCDM fiducial (dark_energy_model = ppf)
)

# CAMB extra_args, verbatim from the config theory block.
CAMB_ARGS = dict(
    dark_energy_model="ppf",
    AccuracyBoost=1.05,
    k_per_logint=10,
    kmax=10.0,                      # theory.camb.extra_args.kmax
    accurate_massive_neutrino_transfers=False,
)
KMAX_BOLTZMANN = 7.5                # likelihood.kmax_boltzmann (P(k) valid range)

REDSHIFTS = [0.0, 0.5, 1.0, 2.0]    # spans where the lensing kernel has weight
KMIN, KMAX_PLOT, NK = 1e-3, 10.0, 600   # h/Mpc for the plotted ratio


def physical_densities(f):
    """Reproduce the config's omegabh2 / omegach2 derived-parameter formulas."""
    h2 = (f["H0"] / 100.0) ** 2
    ombh2 = f["omegab"] * h2
    # config: (omegam - omegab)*(H0/100)^2 - (mnu*(3.046/3)^0.75)/94.0708
    omch2 = (f["omegam"] - f["omegab"]) * h2 - (f["mnu"] * (3.046 / 3.0) ** 0.75) / 94.0708
    return ombh2, omch2


def pk_for(halofit_version, f):
    """Nonlinear P(k,z) for one prescription. Returns (kh, z_array, pk[z,k])."""
    ombh2, omch2 = physical_densities(f)
    pars = camb.set_params(
        H0=f["H0"], ombh2=ombh2, omch2=omch2,
        mnu=f["mnu"], tau=f["tau"], omk=0.0,
        As=f["As_1e9"] * 1e-9, ns=f["ns"],
        num_massive_neutrinos=1, nnu=3.046,
        **CAMB_ARGS,
    )
    # dark energy: ppf with (w0, wa); LCDM at the fiducial
    pars.set_dark_energy(w=f["w0"], wa=f["wa"], dark_energy_model="ppf")
    pars.set_matter_power(redshifts=REDSHIFTS, kmax=KMAX_BOLTZMANN)
    pars.NonLinear = camb.model.NonLinear_both
    pars.NonLinearModel.set_params(halofit_version=halofit_version)
    results = camb.get_results(pars)
    kh, z, pk = results.get_matter_power_spectrum(
        minkh=KMIN, maxkh=KMAX_PLOT, npoints=NK)
    return kh, np.array(z), pk


def main():
    kh, z_taka, pk_taka = pk_for("takahashi", FIDUCIAL)
    _,  z_mead, pk_mead = pk_for("mead2020", FIDUCIAL)
    assert np.allclose(z_taka, z_mead)

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for i, zi in enumerate(z_taka):
        ratio = pk_mead[i] / pk_taka[i]
        ax.plot(kh, ratio, label=f"$z={zi:.1f}$", lw=1.8)

    ax.axhline(1.0, color="0.6", lw=0.8, ls="--", zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(KMIN, KMAX_PLOT)
    ax.set_xlabel(r"$k\;[h\,\mathrm{Mpc}^{-1}]$")
    ax.set_ylabel(r"$P_{\mathrm{Mead}}(k)\,/\,P_{\mathrm{Taka}}(k)$")
    ax.legend(frameon=False, ncol=2, fontsize=9)
    ax.tick_params(direction="in", which="both", top=True, right=True)
    fig.tight_layout()

    # numbers for the caption / a sanity check against the text
    print("k [h/Mpc]   ratio(z=0)   ratio(z=1)")
    for ktarget in (0.1, 0.3, 1.0, 3.0):
        j = np.argmin(np.abs(kh - ktarget))
        r0 = pk_mead[list(z_taka).index(0.0)][j] / pk_taka[list(z_taka).index(0.0)][j]
        r1 = pk_mead[list(z_taka).index(1.0)][j] / pk_taka[list(z_taka).index(1.0)][j]
        print(f"{kh[j]:8.3f}   {r0:9.4f}   {r1:9.4f}")

    out = "figures/pk_ratio_mead_taka.pdf"
    import os
    os.makedirs("figures", exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
