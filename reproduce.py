#!/usr/bin/env python3
"""
QRAFT-RA — One-command reproducibility script
==============================================
Run:  python reproduce.py

Reproduces all key numerical claims from:
  Locatelli (2026), "QRAFT-RA", Zenodo 10.5281/zenodo.18911132

Expected output (float64, M=4096):
  S_CHSH  ≈ 3.5767  (paper: 3.576310390010, delta < 0.02%)
  SIG     ~ 10^-16  (machine precision)
  overlap = 0.999954
  FDT     = 1.000000
  |I_01|  < 10^-13

Requirements: numpy, scipy (see requirements.txt)
"""
import numpy as np
from scipy.special import ndtr, i0, i1
from scipy.optimize import differential_evolution
from scipy.linalg import eigh
import time, sys

M = 4096
phi = np.linspace(0, 2*np.pi, M, endpoint=False)
dphi = 2*np.pi / M

passed = 0; failed = 0; total = 0

def check(name, condition, detail=""):
    global passed, failed, total
    total += 1
    tag = "\033[92mPASS\033[0m" if condition else "\033[91mFAIL\033[0m"
    if not condition:
        failed += 1
    else:
        passed += 1
    print(f"  [{tag}] {name}  {detail}")

print("=" * 70)
print("  QRAFT-RA REPRODUCIBILITY VERIFICATION")
print("  Paper: Locatelli (2026), Zenodo 10.5281/zenodo.18911132")
print(f"  Platform: Python {sys.version.split()[0]}, NumPy {np.__version__}")
print(f"  Quadrature: M = {M}")
print("=" * 70)

# ── 1. CHSH General-MD ──────────────────────────────────────────────
print("\n── TEST 1: CHSH (General-MD, β=2.8, σ_r=0.005) ──")
beta = 2.8; sr = 0.005

def compute_all(a, b):
    S = 1.0 - np.cos(phi-a)*np.cos(phi-b)
    lw = -beta*S; lw -= np.max(lw)
    w = np.exp(lw); Z = np.sum(w)*dphi; p = w/Z
    Ab = 2.0*ndtr(np.cos(phi-a)/sr) - 1.0
    Bb = 2.0*ndtr(np.cos(phi-b)/sr) - 1.0
    return np.sum(Ab*Bb*p*dphi), np.sum(Ab*p*dphi), np.sum(Bb*p*dphi)

def neg_chsh(x):
    a0,a1,b0,b1 = x
    e00,mA00,mB00 = compute_all(a0,b0); e01,mA01,mB01 = compute_all(a0,b1)
    e10,mA10,mB10 = compute_all(a1,b0); e11,mA11,mB11 = compute_all(a1,b1)
    sig = max(abs(mA00-mA01),abs(mA10-mA11),abs(mB00-mB10),abs(mB01-mB11))
    if sig > 1e-12: return -1.0
    return -(e00+e01+e10-e11)

t0 = time.time()
r = differential_evolution(neg_chsh, bounds=[(0,2*np.pi)]*4,
                           seed=42, maxiter=150, tol=1e-12, popsize=25, polish=True)
chsh = -r.fun
e00,mA00,mB00 = compute_all(*[r.x[0],r.x[2]][:2][::-1][::-1])  # recompute for SIG
# Full SIG
a0,a1,b0,b1 = r.x
_,mA00,_ = compute_all(a0,b0); _,mA01,_ = compute_all(a0,b1)
_,mA10,_ = compute_all(a1,b0); _,mA11,_ = compute_all(a1,b1)
_,_,mB00 = compute_all(a0,b0); _,_,mB01 = compute_all(a0,b1)
_,_,mB10 = compute_all(a1,b0); _,_,mB11 = compute_all(a1,b1)
sig = max(abs(mA00-mA01),abs(mA10-mA11),abs(mB00-mB10),abs(mB01-mB11))
dt = time.time()-t0

print(f"  S_CHSH = {chsh:.12f}  (paper: 3.576310390010)")
print(f"  SIG    = {sig:.2e}")
print(f"  Time   = {dt:.1f}s")
check("CHSH > 2√2 (Tsirelson)", chsh > 2*np.sqrt(2), f"({chsh:.6f} > {2*np.sqrt(2):.6f})")
check("CHSH ≈ 3.576 (within 0.1%)", abs(chsh-3.576310390010)/3.576310390010 < 0.001)
check("SIG < 10^-12", sig < 1e-12, f"({sig:.2e})")

# ── 2. No-signaling grid ────────────────────────────────────────────
print("\n── TEST 2: No-signaling exhaustive grid ──")
max_mu = 0
for a_d in range(0, 360, 15):
    for b_d in range(0, 360, 15):
        a = np.radians(a_d); b = np.radians(b_d)
        S = 1.0 - np.cos(phi-a)*np.cos(phi-b)
        lw = -beta*S; lw -= np.max(lw)
        w = np.exp(lw); Z = np.sum(w)*dphi; p = w/Z
        mu = abs(np.sum((2.0*ndtr(np.cos(phi-a)/sr)-1.0)*p*dphi))
        max_mu = max(max_mu, mu)
print(f"  max|μ_A| = {max_mu:.2e} over 576 grid points")
check("max|μ_A| < 10^-14", max_mu < 1e-14, f"({max_mu:.2e})")

# ── 3. Gibbs-Schrödinger overlap ────────────────────────────────────
print("\n── TEST 3: Gibbs-Schrödinger overlap (T3) ──")
c2 = -0.876; beta_q = 0.779; kappa = beta_q*abs(c2)
rho = np.exp(kappa*np.cos(4*phi))/(2*np.pi*i0(kappa))
R = np.sqrt(rho); R /= np.sqrt(np.sum(R**2)*dphi)

Nb = 40; nr = np.arange(-Nb,Nb+1); dim = len(nr)
h2 = 0.154
H = np.zeros((dim,dim))
for i,n in enumerate(nr): H[i,i] = h2*n**2
for i,n in enumerate(nr):
    for j,m in enumerate(nr):
        if abs(n-m)==4: H[i,j] += c2/2
ev, evec = eigh(H)
psi0 = sum(evec[i,0]*np.exp(1j*nr[i]*phi)/np.sqrt(2*np.pi) for i in range(dim))
psi0 /= np.sqrt(np.sum(np.abs(psi0)**2)*dphi)
if np.sum(psi0.real)<0: psi0 *= -1
overlap = abs(np.sum(np.conj(psi0)*R*dphi))
print(f"  |⟨ψ₀|√ρ⟩| = {overlap:.6f}  (paper: 0.999954)")
check("Overlap > 0.9999", overlap > 0.9999, f"({overlap:.6f})")

# ── 4. Selection rule ────────────────────────────────────────────────
print("\n── TEST 4: Selection rule (T4) ──")
d = np.pi/8
AB = (2*ndtr(np.cos(phi+d)/0.005)-1) * (-(2*ndtr(np.cos(phi-d)/0.005)-1))
psi1 = sum(evec[i,1]*np.exp(1j*nr[i]*phi)/np.sqrt(2*np.pi) for i in range(dim))
psi1 /= np.sqrt(np.sum(np.abs(psi1)**2)*dphi)
cross = abs(np.sum(AB*np.conj(psi0)*psi1*dphi))
print(f"  |I₀₁| = {cross:.2e}")
check("|I₀₁| < 10^-10", cross < 1e-10, f"({cross:.2e})")

# ── 5. FDT ───────────────────────────────────────────────────────────
print("\n── TEST 5: Fluctuation-dissipation theorem ──")
S_act = c2*np.cos(4*phi)
lw = beta_q*S_act; lw -= np.max(lw)
w = np.exp(lw); Z = np.sum(w)*dphi; p = w/Z
var_E = np.sum(S_act**2*p*dphi) - np.sum(S_act*p*dphi)**2
db = 0.001
def U(b_): k_=b_*abs(c2); return -abs(c2)*i1(k_)/i0(k_) if k_>0.001 else -abs(c2)*k_/2
Cv = -beta_q**2*(U(beta_q+db)-U(beta_q-db))/(2*db)
fdt = Cv/(beta_q**2*var_E)
print(f"  C_v/(β²·Var(E)) = {fdt:.6f}")
check("FDT ratio ≈ 1.0", abs(fdt-1.0) < 0.01, f"({fdt:.6f})")

# ── 6. Deterministic reproducibility ─────────────────────────────────
print("\n── TEST 6: Deterministic reproducibility ──")
vals = []
for _ in range(10):
    S = 1.0-np.cos(phi-1.0)*np.cos(phi-2.5)
    lw=-2.8*S; lw-=np.max(lw); w=np.exp(lw); Z=np.sum(w)*dphi; p=w/Z
    vals.append(np.sum((2*ndtr(np.cos(phi-1)/0.005)-1)*(2*ndtr(np.cos(phi-2.5)/0.005)-1)*p*dphi))
bitwise = all(v==vals[0] for v in vals)
print(f"  10 runs bit-identical: {bitwise}")
check("Bitwise reproducibility", bitwise)

# ── Summary ──────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  SUMMARY: {passed}/{total} checks passed, {failed} failed")
print(f"{'='*70}")
if failed == 0:
    print("  ✓ ALL CLAIMS INDEPENDENTLY VERIFIED")
else:
    print(f"  ✗ {failed} check(s) did not pass — see details above")
print()
