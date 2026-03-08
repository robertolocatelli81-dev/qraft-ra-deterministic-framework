# QRAFT-RA Deterministic Framework

[

![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18911132.svg)

](https://doi.org/10.5281/zenodo.18911132)
[

![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)

](https://creativecommons.org/licenses/by/4.0/)
[

![Reproduce](https://img.shields.io/badge/reproduce.py-8%2F8_PASS-brightgreen.svg)

]()

**Quadrature-based Reproducible Action-Structured Framework with Regularized Action**

Deterministic contextual variational framework for generating and verifying non-classical correlations through a fully reproducible computational pipeline.

> **Author:** Roberto Locatelli
> **Paper:** [Zenodo 10.5281/zenodo.18911132](https://doi.org/10.5281/zenodo.18911132)
> **Patent:** PCT/IB2026/050200

---

## Quick Start

```bash
git clone https://github.com/robertolocatelli81-dev/qraft-ra-deterministic-framework.git
cd qraft-ra-deterministic-framework
pip install -r requirements.txt
python reproduce.py
Output (~35 seconds):
[PASS] CHSH > 2√2 (Tsirelson)       (3.576699 > 2.828427)
[PASS] CHSH ≈ 3.576 (within 0.1%)
[PASS] SIG < 10^-12                  (8.33e-17)
[PASS] max|μ_A| < 10^-14             (4.44e-16)
[PASS] Overlap > 0.9999              (0.999954)
[PASS] |I₀₁| < 10^-10               (3.56e-14)
[PASS] FDT ratio ≈ 1.0               (1.000000)
[PASS] Bitwise reproducibility

SUMMARY: 8/8 checks passed, 0 failed
✓ ALL CLAIMS INDEPENDENTLY VERIFIED
Key Results
Quantity
Value
Description
S_CHSH
3.576310390010
Super-Tsirelson CHSH (General-MD regime)
SIG
~10⁻¹⁶
Operational no-signaling (machine precision)
⟨ψ₀|√ρ⟩
0.999954
Gibbs–Schrödinger ground state overlap
FDT
1.000000
Fluctuation-dissipation ratio
Selection rule
< 10⁻¹⁵
Harmonic cross-term (identically zero)
Repository Structure
reproduce.py                  ← One-command verification (run this first)
requirements.txt              ← Dependencies: numpy, scipy, sympy

code/                         ← 14 physics scripts (all paper claims)
├── nosig_derivazione_completa.py   T1: No-signaling analytic proof
├── decisive_test.py                T3: ψ = √ρ · exp(iS/ℏ) overlap
├── theorems_and_wave.py            T1–T5 + canonical quantization
├── symmetry_breaking.py            T4: Parity/symmetry breaking
├── fourier_bell_memory.py          Fourier-Bell connection + memory
├── quadrupole_origin.py            T2: Why frequency 4 dominates
├── harmonics_study.py              Spectral analysis
├── gap_to_qm.py                   β → QM transition
├── spin32.py                       Generalization to arbitrary spin
├── excited_state.py                First excited state
└── missing_patterns.py             Parisi-Wu, Wigner, Bohm connections

reproducibility/              ← 7 stress tests and cross-checks
├── testA1.py                       Spin ½ vs photons
├── testB.py                        Full quantum phenomenology
├── testC.py                        Experimental predictions
├── stress_final_1.py               Thermodynamics + FDT
├── stress_final_2.py               Multipartite limitations
└── reproduce_discrepancy.py        RMSE discrepancy diagnosis

paper/                        ← Paper + verification report
How It Works
QRAFT-RA constructs a contextual action on a compact latent domain:
S(φ; a, b) = 1 − cos(φ − a) cos(φ − b)
A Gibbs distribution p(φ|a,b;β) ∝ exp[−β S] controls a noise-aware measurement map, producing deterministic correlators via quadrature integration. No Monte Carlo sampling is involved.
Citation
@misc{locatelli2026qraftra,
  author       = {Locatelli, Roberto},
  title        = {{QRAFT-RA}: Quadrature-based Reproducible Action-Structured
                  Framework with Regularized Action},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18911132},
  url          = {https://doi.org/10.5281/zenodo.18911132}
}
License
CC-BY 4.0
Disclaimer
This framework is presented as a deterministic mathematical and computational archetype. It does not claim microscopic physical realizability. The super-Tsirelson CHSH value arises from measurement dependence in the contextual Gibbs distribution.
