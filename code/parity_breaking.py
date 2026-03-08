"""
═══════════════════════════════════════════════════════════════════════════
  ROTTURA DI PARITÀ: ε·sin(φ)
  
  Ĥ(ε) = −(ℏ²/2m)∂² + c₂cos(4φ) + ε·sin(φ)
  
  sin(−φ) = −sin(φ) → H(−φ) ≠ H(φ) → parità ROTTA
  → ψ₀ e ψ₁ non hanno più parità definita
  → il cross-term ψ₀*ψ₁ non è più puramente dispari
  → interferenza possibile
═══════════════════════════════════════════════════════════════════════════
"""
import numpy as np
from scipy.special import ndtr, i0
from scipy.linalg import eigh

M = 2048
phi = np.linspace(0, 2*np.pi, M, endpoint=False)
dp = 2*np.pi / M
c2 = -0.876; h2_2m = 0.154; sr = 0.005
kappa = 0.779 * abs(c2)

N_basis = 40
n_range = np.arange(-N_basis, N_basis+1)
dim = len(n_range)

def build_H(epsilon):
    """Ĥ = T + c₂cos(4φ) + ε·sin(φ)"""
    H = np.zeros((dim, dim), dtype=complex)
    for i, n in enumerate(n_range):
        H[i, i] = h2_2m * n**2
    for i, n in enumerate(n_range):
        for j, m in enumerate(n_range):
            if abs(n-m) == 4: H[i,j] += c2/2
            # sin(φ) = (e^(iφ) − e^(−iφ))/(2i)
            if n-m == 1:  H[i,j] += epsilon/(2j)   # e^(iφ) term
            if n-m == -1: H[i,j] -= epsilon/(2j)   # e^(-iφ) term
    # H è hermitiana? sin(φ) nel kernel: ⟨n|sin(φ)|m⟩ = (δ_{n,m+1} - δ_{n,m-1})/(2i)
    # Correggiamo: ⟨n|sin(φ)|m⟩ = -i/2 (δ_{n,m+1}) + i/2 (δ_{n,m-1})
    H2 = np.zeros((dim, dim), dtype=complex)
    for i, n in enumerate(n_range):
        H2[i, i] = h2_2m * n**2
    for i, n in enumerate(n_range):
        for j, m in enumerate(n_range):
            if abs(n-m) == 4: H2[i,j] += c2/2
            if n-m == 1:  H2[i,j] += -1j*epsilon/2
            if n-m == -1: H2[i,j] += 1j*epsilon/2
    return H2

def build_psi(eigvec):
    psi = np.zeros(M, dtype=complex)
    for i, n in enumerate(n_range):
        psi += eigvec[i] * np.exp(1j*n*phi) / np.sqrt(2*np.pi)
    norm = np.sqrt(np.sum(np.abs(psi)**2)*dp)
    return psi/norm

def correlator(prob, dtheta):
    d = dtheta/2
    A = 2*ndtr(np.cos(phi+d)/sr)-1
    B = -(2*ndtr(np.cos(phi-d)/sr)-1)
    return np.sum(A*B*prob*dp)

rho_gibbs = np.exp(kappa*np.cos(4*phi))/(2*np.pi*i0(kappa))
R_gibbs = np.sqrt(rho_gibbs)
R_gibbs_norm = R_gibbs/np.sqrt(np.sum(R_gibbs**2)*dp)

print("=" * 78)
print("  ROTTURA DI PARITÀ: Ĥ = Ĥ₀ + ε·sin(φ)")
print("  sin(−φ) = −sin(φ) → parità ROTTA")
print("=" * 78)

# Verifica hermitianità
H_test = build_H(0.1)
herm_err = np.max(np.abs(H_test - H_test.conj().T))
print(f"\n  Verifica: max|H − H†| = {herm_err:.2e} ({'OK' if herm_err < 1e-12 else 'ERRORE'})")

# ═══════════════════════════════════════════════════════════════════════
# 1. SPETTRO
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 78}")
print("  1. SPETTRO CON ROTTURA DI PARITÀ")
print("─" * 78)

print(f"  {'ε':>6} {'E₀':>10} {'E₁':>10} {'E₂':>10} {'Gap₀₁':>8} {'Gap₁₂':>8}")
print(f"  {'─'*50}")

for eps in [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
    H = build_H(eps)
    evals = eigh(H, eigvals_only=True)
    print(f"  {eps:6.3f} {evals[0]:10.6f} {evals[1]:10.6f} {evals[2]:10.6f} "
          f"{evals[1]-evals[0]:8.4f} {evals[2]-evals[1]:8.6f}")

# ═══════════════════════════════════════════════════════════════════════
# 2. PARITÀ DEGLI AUTOSTATI
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 78}")
print("  2. PARITÀ DEGLI AUTOSTATI (CONTENUTO PARI vs DISPARI)")
print("─" * 78)

print(f"  {'ε':>6} {'P₀(pari)':>10} {'P₁(pari)':>10} {'P₂(pari)':>10}")
print(f"  {'─'*40}")

for eps in [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
    H = build_H(eps)
    evals, evecs = eigh(H)
    
    for state_idx in range(3):
        psi = build_psi(evecs[:, state_idx])
        # Parità: ψ(−φ) = ψ(2π−φ)
        psi_flip = np.zeros(M, dtype=complex)
        for i, n in enumerate(n_range):
            psi_flip += evecs[state_idx, i] * np.exp(-1j*n*phi) / np.sqrt(2*np.pi)
        psi_flip /= np.sqrt(np.sum(np.abs(psi_flip)**2)*dp)
        
        # Proiezione pari: (ψ + ψ_flip)/2
        psi_even = (psi + psi_flip)/2
        frac_even = np.sum(np.abs(psi_even)**2*dp) / np.sum(np.abs(psi)**2*dp)
        
        if state_idx == 0:
            print(f"  {eps:6.3f}", end="")
        print(f" {frac_even:10.4f}", end="")
    print()

print("""
  ★ A ε=0: ψ₀ è 100% pari, ψ₁ è 0% pari (100% dispari).
  Con ε>0: la parità si MESCOLA — gli stati non sono più puri.
  Questo è il prerequisito per interferenza non-nulla.
""")

# ═══════════════════════════════════════════════════════════════════════
# 3. IL CROSS-TERM DI INTERFERENZA
# ═══════════════════════════════════════════════════════════════════════
print(f"{'─' * 78}")
print("  3. CROSS-TERM DI INTERFERENZA vs ε (Δθ = 45°, 60°, 90°)")
print("─" * 78)

angles_test = [45, 60, 90]
AB_cache = {}
for deg in angles_test:
    d = np.radians(deg)/2
    A = 2*ndtr(np.cos(phi+d)/sr)-1
    B = -(2*ndtr(np.cos(phi-d)/sr)-1)
    AB_cache[deg] = A*B

print(f"  α = β = 1/√2")
print(f"\n  {'ε':>6}", end="")
for deg in angles_test:
    print(f"  {'I('+str(deg)+'°)':>12}", end="")
print(f"  {'max|Interf|':>12}")
print(f"  {'─'*56}")

results = {}
for eps in [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
    H = build_H(eps)
    evals, evecs = eigh(H)
    
    psi0 = build_psi(evecs[:,0])
    psi1 = build_psi(evecs[:,1])
    
    # Fissare fase globale
    if np.sum(np.abs(psi0.real)) < np.sum(np.abs(psi0.imag)):
        psi0 *= 1j
    if psi0[np.argmax(np.abs(psi0))].real < 0:
        psi0 *= -1
    
    psi_s = (psi0 + psi1)/np.sqrt(2)
    psi_s /= np.sqrt(np.sum(np.abs(psi_s)**2)*dp)
    prob_mix = 0.5*np.abs(psi0)**2 + 0.5*np.abs(psi1)**2
    
    print(f"  {eps:6.3f}", end="")
    max_int = 0
    interf_dict = {}
    
    for deg in angles_test:
        AB = AB_cache[deg]
        E_qm = np.sum(AB * np.abs(psi_s)**2 * dp)
        E_mix = np.sum(AB * prob_mix * dp)
        interf = E_qm - E_mix
        interf_dict[deg] = interf
        max_int = max(max_int, abs(interf))
        print(f"  {interf:+12.6f}", end="")
    
    print(f"  {max_int:12.6f}")
    results[eps] = (max_int, interf_dict)

# ═══════════════════════════════════════════════════════════════════════
# 4. SWEEP ANGOLARE COMPLETO A ε OTTIMALE
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 78}")
print("  4. SWEEP ANGOLARE COMPLETO")
print("─" * 78)

# Trova ε con massima interferenza
best_eps = max(results.keys(), key=lambda e: results[e][0])
print(f"  ε con max interferenza: {best_eps}")

for eps_show in [0.1, 0.5, best_eps]:
    H = build_H(eps_show)
    evals, evecs = eigh(H)
    psi0 = build_psi(evecs[:,0])
    psi1 = build_psi(evecs[:,1])
    if psi0[np.argmax(np.abs(psi0))].real < 0: psi0 *= -1
    
    psi_s = (psi0+psi1)/np.sqrt(2)
    psi_s /= np.sqrt(np.sum(np.abs(psi_s)**2)*dp)
    prob_mix = 0.5*np.abs(psi0)**2 + 0.5*np.abs(psi1)**2
    
    print(f"\n  ε = {eps_show}:")
    print(f"  {'Δθ':>5} {'E_QM':>10} {'E_mix':>10} {'Interf':>10} {'|I₀₁|':>10}")
    print(f"  {'─'*48}")
    
    max_i = 0
    for deg in [0,15,30,45,60,75,90,105,120,135,150,165,180]:
        d = np.radians(deg)/2
        A = 2*ndtr(np.cos(phi+d)/sr)-1
        B = -(2*ndtr(np.cos(phi-d)/sr)-1)
        AB = A*B
        
        E_q = np.sum(AB*np.abs(psi_s)**2*dp)
        E_m = np.sum(AB*prob_mix*dp)
        interf = E_q - E_m
        I01 = abs(np.sum(AB*np.conj(psi0)*psi1*dp))
        max_i = max(max_i, abs(interf))
        
        marker = " ←" if abs(interf) > 0.001 else ""
        print(f"  {deg:4d}° {E_q:10.6f} {E_m:10.6f} {interf:+10.6f} {I01:10.6f}{marker}")
    
    print(f"  max|Interferenza| = {max_i:.6f}")

# ═══════════════════════════════════════════════════════════════════════
# 5. CRESCITA DELL'INTERFERENZA CON ε
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 78}")
print("  5. LEGGE DI SCALA: INTERFERENZA vs ε")
print("─" * 78)

eps_vals = []
int_vals = []

for eps in np.concatenate([np.arange(0.001, 0.01, 0.001),
                            np.arange(0.01, 0.1, 0.01),
                            np.arange(0.1, 1.01, 0.1)]):
    H = build_H(eps)
    evals, evecs = eigh(H)
    psi0 = build_psi(evecs[:,0])
    psi1 = build_psi(evecs[:,1])
    if psi0[np.argmax(np.abs(psi0))].real < 0: psi0 *= -1
    
    psi_s = (psi0+psi1)/np.sqrt(2)
    psi_s /= np.sqrt(np.sum(np.abs(psi_s)**2)*dp)
    prob_mix = 0.5*np.abs(psi0)**2 + 0.5*np.abs(psi1)**2
    
    # Max interferenza su tutti gli angoli
    max_int = 0
    for deg in range(0, 181, 15):
        d = np.radians(deg)/2
        A = 2*ndtr(np.cos(phi+d)/sr)-1
        B = -(2*ndtr(np.cos(phi-d)/sr)-1)
        E_q = np.sum(A*B*np.abs(psi_s)**2*dp)
        E_m = np.sum(A*B*prob_mix*dp)
        max_int = max(max_int, abs(E_q-E_m))
    
    eps_vals.append(eps)
    int_vals.append(max_int)

eps_arr = np.array(eps_vals)
int_arr = np.array(int_vals)

# Mostra campione
print(f"  {'ε':>8} {'max|Interf|':>12} {'log₁₀(|I|)':>12}")
print(f"  {'─'*36}")
for i in range(0, len(eps_arr), max(1, len(eps_arr)//15)):
    log_i = np.log10(int_arr[i]) if int_arr[i] > 1e-15 else -15
    print(f"  {eps_arr[i]:8.4f} {int_arr[i]:12.6f} {log_i:12.2f}")

# Fit power law per ε piccoli
mask = (eps_arr > 0.005) & (eps_arr < 0.2) & (int_arr > 1e-12)
if np.sum(mask) > 2:
    log_e = np.log10(eps_arr[mask])
    log_i = np.log10(int_arr[mask] + 1e-15)
    coeffs = np.polyfit(log_e, log_i, 1)
    alpha = coeffs[0]
    print(f"\n  Fit: |Interferenza| ∝ ε^{alpha:.2f} (regime perturbativo)")
    if abs(alpha - 1) < 0.3:
        print(f"  → LINEARE in ε (primo ordine)")
    elif abs(alpha - 2) < 0.3:
        print(f"  → QUADRATICA in ε (secondo ordine)")
    else:
        print(f"  → Esponente {alpha:.2f}")

# ═══════════════════════════════════════════════════════════════════════
# 6. VERDETTO
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 78}")
print("  VERDETTO: ROTTURA DI PARITÀ CON sin(φ)")
print("═" * 78)

max_ever = max(int_arr)
eps_at_max = eps_arr[np.argmax(int_arr)]

emerged = max_ever > 0.001

print(f"""
  RISULTATO:
  
  max|Interferenza| = {max_ever:.6f} a ε = {eps_at_max:.3f}
  
  {'★ IL CROSS-TERM È EMERSO!' if emerged else '✗ Il cross-term è ancora nullo.'}
  
  {'L interferenza quantistica tra ψ₀ e ψ₁ diventa VISIBILE' if emerged else 'L interferenza resta sotto la soglia di rilevazione'}
  {'quando la parità del mezzo è rotta da ε·sin(φ).' if emerged else 'anche con rottura di parità.'}
  
  Interpretazione:
  {'Con parità intatta (cos φ): selection rule proibisce I₀₁ → zero esatto.' if True else ''}
  {'Con parità rotta (sin φ): gli stati si mescolano → I₀₁ ≠ 0.' if emerged else 'Serve investigare ulteriormente.'}
  
  {'L interferenza cresce come ε^α con α ≈ ' + f'{alpha:.1f}' if emerged and 'alpha' in dir() else ''}
  
  ★ SIGNIFICATO PER IL MODELLO VIBRAZIONALE:
  
  {'Il mezzo vibrazionale ha settori di simmetria protetti.' if True else ''}
  {'L interferenza quantistica è nascosta da selection rules,' if True else ''}
  {'non assente. Basta rompere la simmetria giusta per rivelarla.' if emerged else ''}
  {'Questo è esattamente il comportamento di un sistema quantistico reale.' if emerged else ''}
""")

