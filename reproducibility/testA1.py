"""TEST A1: Confronto con forma sperimentale dei fotoni"""
import numpy as np
from scipy.special import ndtr, i0
from scipy.optimize import differential_evolution

M = 4096
psi = np.linspace(0, 2*np.pi, M, endpoint=False)
dp = 2*np.pi / M
sr = 0.005

def E_model(dt, c1, c2, c3, beta):
    S = c1*np.cos(2*psi) + c2*np.cos(4*psi) + c3*np.cos(6*psi)
    lw = beta*S; lw -= np.max(lw)
    w = np.exp(lw); Z = np.sum(w)*dp; p = w/Z
    d = dt/2
    A = 2*ndtr(np.cos(psi+d)/sr)-1
    B = -(2*ndtr(np.cos(psi-d)/sr)-1)
    return np.sum(A*B*p*dp)

# Ottimo quadrupolare precedente
c1o, c2o, c3o, bo = 0.0, -0.876, 0.0, 0.779

print("=" * 78)
print("  A1: SPIN ½ vs SPIN 1 (FOTONI)")
print("=" * 78)
print("""
  Esperimenti Bell con fotoni: E_exp = −cos(2Δθ)  [spin 1]
  Spin ½ (elettroni):          E_exp = −cos(Δθ)
  
  Il fattore 2 viene dalla rappresentazione dello spin.
""")

print(f"  {'Δθ':>6} {'Modello':>10} {'−cos(Δθ)':>10} {'−cos(2Δθ)':>10}")
print(f"  {'─'*40}")

err1, err2, n = 0, 0, 0
for deg in [0,15,30,45,60,75,90,120,135,150,180]:
    d = np.radians(deg)
    Em = E_model(d, c1o, c2o, c3o, bo)
    E1 = -np.cos(d); E2 = -np.cos(2*d)
    err1 += (Em-E1)**2; err2 += (Em-E2)**2; n += 1
    print(f"  {deg:5d}° {Em:10.4f} {E1:10.4f} {E2:10.4f}")

print(f"\n  RMSE vs −cos(Δθ):  {np.sqrt(err1/n):.4f}  (spin ½)")
print(f"  RMSE vs −cos(2Δθ): {np.sqrt(err2/n):.4f}  (fotoni)")
print(f"\n  ★ Il modello attuale approssima spin ½, NON fotoni.")
print(f"    Per fotoni serve raddoppiare la frequenza della mappa di misura.")

# Test mappa fotonica: cos(2(φ−θ))
print(f"\n{'─' * 78}")
print("  A2: MAPPA FOTONICA cos(2(φ−θ)) + distribuzione π/2-periodica")
print("─" * 78)

def E_photon(dt, coeffs_4n, beta):
    S = sum(cn*np.cos(4*n*psi) for n,cn in enumerate(coeffs_4n,1))
    lw = beta*S; lw -= np.max(lw)
    w = np.exp(lw); Z = np.sum(w)*dp; p = w/Z
    d = dt/2
    A = 2*ndtr(np.cos(2*(psi+d))/sr)-1
    B = -(2*ndtr(np.cos(2*(psi-d))/sr)-1)
    return np.sum(A*B*p*dp)

def rmse_ph(params):
    c1,c2,beta = params
    if beta < 0.01: return 100
    e = sum((E_photon(np.radians(deg),[c1,c2],beta) - (-np.cos(np.radians(2*deg))))**2 
            for deg in range(0,91,3))
    return np.sqrt(e/31)

print("  Ottimizzazione...")
res = differential_evolution(rmse_ph, bounds=[(-2,2),(-2,2),(0.01,10)],
                              seed=42, maxiter=80, tol=1e-10, popsize=15)
c4, c8, bp = res.x

print(f"  c_4={c4:.4f}, c_8={c8:.4f}, β={bp:.4f}, RMSE={res.fun:.6f}")

# NS check
max_mu = 0
for deg in range(0,360,10):
    d = np.radians(deg)/2
    S = c4*np.cos(4*psi) + c8*np.cos(8*psi)
    lw = bp*S; lw -= np.max(lw)
    w = np.exp(lw); Z = np.sum(w)*dp; p = w/Z
    A = 2*ndtr(np.cos(2*(psi+d))/sr)-1
    max_mu = max(max_mu, abs(np.sum(A*p*dp)))

print(f"  No-signaling: max|μ| = {max_mu:.2e} → {'OK' if max_mu<1e-10 else 'FAIL'}")

# La condizione NS per mappa fotonica:
# f(ψ) = 2Φ(cos(2(ψ+δ))/σ) - 1
# f(ψ+π/2) = 2Φ(cos(2ψ+2δ+π))/σ) - 1 = 2Φ(-cos(2ψ+2δ)/σ) - 1 = -f(ψ) ✓
# Serve g con periodo π/2: cos(4nψ) ✓

print(f"\n  Confronto modello fotonico vs −cos(2Δθ):")
print(f"  {'Δθ':>6} {'E_fot':>10} {'−cos(2Δθ)':>10} {'Errore':>10}")
print(f"  {'─'*40}")
for deg in [0,10,15,20,22.5,30,45,60,67.5,75,80,90]:
    d = np.radians(deg)
    Em = E_photon(d, [c4,c8], bp)
    Eq = -np.cos(2*d)
    print(f"  {deg:6.1f} {Em:10.6f} {Eq:10.6f} {abs(Em-Eq):10.6f}")

print(f"""
  ★ VERDETTO A:
  1) Il modello BASE (mappa cos(φ−θ)) descrive spin ½, non fotoni
  2) Con mappa cos(2(φ−θ)) e distribuzione π/2-periodica → descrive fotoni
  3) Il no-signaling richiede armoniche cos(4nψ) per la mappa fotonica
  4) La struttura è GENERALIZZABILE: spin s → mappa cos(2s(φ−θ)), 
     distribuzione con periodo π/(2s), armoniche cos(4snψ)
""")
