"""
VERIFICA ONESTA: dove il 0.027 è stato ottenuto e perché 
la riproduzione diretta dà 0.16-0.28
"""
import numpy as np
from scipy.special import ndtr, i0
from scipy.optimize import differential_evolution

M = 2048
phi = np.linspace(0, 2*np.pi, M, endpoint=False)
dp = 2*np.pi / M

def E_corr(delta_theta, c1, c2, beta, sigma):
    """Correlatore con S = c1*cos(2ψ) + c2*cos(4ψ)"""
    d = delta_theta / 2
    S = c1*np.cos(2*phi) + c2*np.cos(4*phi)
    lw = beta*S; lw -= np.max(lw)
    w = np.exp(lw); Z = np.sum(w)*dp; rho = w/Z
    A = 2*ndtr(np.cos(phi+d)/sigma)-1
    B = -(2*ndtr(np.cos(phi-d)/sigma)-1)
    return np.sum(A*B*rho*dp)

def rmse_vs_singlet(c1, c2, beta, sigma):
    err = 0; n = 0
    for deg in range(0, 181, 5):
        dt = np.radians(deg)
        E = E_corr(dt, c1, c2, beta, sigma)
        err += (E - (-np.cos(dt)))**2; n += 1
    return np.sqrt(err/n)

print("=" * 78)
print("  VERIFICA ONESTA: TRACCIAMENTO DEL RMSE = 0.027")
print("=" * 78)

# Test 1: parametri dichiarati nel paper
print("\n  TEST 1: c1=0, c2=-0.876, β=0.779, σ=0.005")
r1 = rmse_vs_singlet(0, -0.876, 0.779, 0.005)
print(f"  RMSE = {r1:.6f}")

# Test 2: ottimizzazione globale fresca
print("\n  TEST 2: Ottimizzazione globale (differential evolution)")
def obj(params):
    c1, c2, beta = params
    if beta < 0.01: return 10
    return rmse_vs_singlet(c1, c2, beta, 0.005)

res = differential_evolution(obj, [(-2,2),(-2,0),(0.01,5)], 
                              seed=42, maxiter=100, tol=1e-10, popsize=20)
print(f"  c1={res.x[0]:.4f}, c2={res.x[1]:.4f}, β={res.x[2]:.4f}")
print(f"  RMSE = {res.fun:.6f}")

# Test 3: con sigma diversi
print("\n  TEST 3: Effetto di σ all'ottimo")
c1o, c2o, bo = res.x
for sigma in [0.2, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001]:
    r = rmse_vs_singlet(c1o, c2o, bo, sigma)
    print(f"  σ={sigma:.4f}: RMSE = {r:.6f}")

# Test 4: correlatore punto per punto all'ottimo
print(f"\n  TEST 4: E(Δθ) all'ottimo (c1={c1o:.3f}, c2={c2o:.3f}, β={bo:.3f}, σ=0.005)")
print(f"  {'Δθ':>5} {'E_mod':>10} {'−cosΔθ':>10} {'Errore':>10}")
print(f"  {'─'*38}")
for deg in [0,15,30,45,60,75,90,120,150,180]:
    dt = np.radians(deg)
    E = E_corr(dt, c1o, c2o, bo, 0.005)
    tgt = -np.cos(dt)
    print(f"  {deg:4d}° {E:10.6f} {tgt:10.6f} {abs(E-tgt):10.6f}")

# Test 5: il vecchio ottimo dal summary era diverso?
# Il summary dice "c₁≈0, c₂≈−0.876, c₃≈0, β≈0.779" -> RMSE 0.027
# Ma con 3 armoniche?
print("\n  TEST 5: Con 3 armoniche (c1, c2, c3)")
def E_corr3(dt, c1, c2, c3, beta, sigma):
    d = dt/2
    S = c1*np.cos(2*phi) + c2*np.cos(4*phi) + c3*np.cos(6*phi)
    lw = beta*S; lw -= np.max(lw)
    w = np.exp(lw); Z = np.sum(w)*dp; rho = w/Z
    A = 2*ndtr(np.cos(phi+d)/sigma)-1
    B = -(2*ndtr(np.cos(phi-d)/sigma)-1)
    return np.sum(A*B*rho*dp)

def obj3(params):
    c1,c2,c3,beta = params
    if beta < 0.01: return 10
    err = 0; n = 0
    for deg in range(0,181,5):
        dt = np.radians(deg)
        E = E_corr3(dt, c1, c2, c3, beta, 0.005)
        err += (E - (-np.cos(dt)))**2; n += 1
    return np.sqrt(err/n)

res3 = differential_evolution(obj3, [(-2,2),(-2,0),(-1,1),(0.01,5)],
                               seed=42, maxiter=100, tol=1e-10, popsize=20)
print(f"  c1={res3.x[0]:.4f}, c2={res3.x[1]:.4f}, c3={res3.x[2]:.4f}, β={res3.x[3]:.4f}")
print(f"  RMSE = {res3.fun:.6f}")

print(f"""
{'═' * 78}
  DIAGNOSI ONESTA
{'═' * 78}

  I parametri c1=0, c2=-0.876, β=0.779 danno RMSE = {r1:.4f}, NON 0.027.
  
  L'ottimizzazione globale fresca trova RMSE = {res.fun:.4f} (2 armoniche)
  e RMSE = {res3.fun:.4f} (3 armoniche).
  
  Il valore 0.027 dichiarato nel paper e nel summary precedente era 
  {'CORRETTO — riprodotto con i parametri ottimali.' if res.fun < 0.03 or res3.fun < 0.03 else 'NON RIPRODUCIBILE con la presente implementazione.'}
  
  {'La discrepanza con 0.027 suggerisce che nell ottimizzazione originale' if res.fun > 0.03 else ''}
  {'c era una definizione diversa del correlatore o parametri aggiuntivi.' if res.fun > 0.03 else ''}
""")

