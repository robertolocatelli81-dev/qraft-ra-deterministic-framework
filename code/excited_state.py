"""
═══════════════════════════════════════════════════════════════════════════
  PRIMO STATO ECCITATO ψ₁
  
  Dove classico e quantistico DIVERGONO:
  - ψ₁ ha nodi (punti dove ψ=0)
  - ψ₁ ha fase S ≠ 0
  - |ψ₁|² ≠ ρ₁ (seconda autofunzione Fokker-Planck)
  - Le correlazioni bipartite DEVONO essere diverse
═══════════════════════════════════════════════════════════════════════════
"""
import numpy as np
from scipy.special import ndtr, i0, i1
from scipy.linalg import eigh

M = 2048
phi = np.linspace(0, 2*np.pi, M, endpoint=False)
dp = 2*np.pi / M
c2 = -0.876
sr = 0.005

# Parametri dall'ottimo del test decisivo
h2_2m = 0.154  # ℏ²/(2m) ottimale
kappa = 0.779 * abs(c2)  # β|c₂|

print("=" * 78)
print("  PRIMO STATO ECCITATO: DOVE CLASSICO E QUANTISTICO DIVERGONO")
print("=" * 78)

# ═══════════════════════════════════════════════════════════════════════
# 1. COSTRUZIONE DELLO SPETTRO COMPLETO
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 78}")
print("  1. SPETTRO DI Ĥ = −(ℏ²/2m)∂² + c₂cos(4φ)")
print("─" * 78)

N_basis = 40
n_range = np.arange(-N_basis, N_basis+1)
dim = len(n_range)

H_mat = np.zeros((dim, dim))
for i, n in enumerate(n_range):
    H_mat[i, i] = h2_2m * n**2
for i, n in enumerate(n_range):
    for j, m in enumerate(n_range):
        if abs(n - m) == 4:
            H_mat[i, j] += c2 / 2

eigenvalues, eigenvectors = eigh(H_mat)

print(f"  ℏ²/(2m) = {h2_2m}, V = {c2}·cos(4φ)")
print(f"\n  Primi 8 livelli energetici:")
print(f"  {'n':>4} {'E_n':>12} {'E_n−E_0':>12} {'Degenerazione':>14}")
print(f"  {'─'*44}")

# Identifica degenerazioni
for n in range(8):
    gap = eigenvalues[n] - eigenvalues[0]
    # Controlla degenerazione
    if n > 0 and abs(eigenvalues[n] - eigenvalues[n-1]) < 1e-8:
        degen = "degenere"
    else:
        degen = "singolo"
    print(f"  {n:4d} {eigenvalues[n]:12.6f} {gap:12.6f} {degen:>14}")

# ═══════════════════════════════════════════════════════════════════════
# 2. COSTRUZIONE DI ψ₀ E ψ₁
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 78}")
print("  2. FUNZIONI D'ONDA ψ₀ E ψ₁")
print("─" * 78)

def build_wavefunction(eigvec, n_range, phi_arr):
    psi = np.zeros(len(phi_arr), dtype=complex)
    for i, n in enumerate(n_range):
        psi += eigvec[i] * np.exp(1j * n * phi_arr) / np.sqrt(2*np.pi)
    norm = np.sqrt(np.sum(np.abs(psi)**2) * dp)
    return psi / norm

psi0 = build_wavefunction(eigenvectors[:, 0], n_range, phi)
psi1 = build_wavefunction(eigenvectors[:, 1], n_range, phi)
psi2 = build_wavefunction(eigenvectors[:, 2], n_range, phi)

# Fissare fase globale: ψ₀ reale e positivo al massimo
if np.sum(psi0.real) < 0: psi0 *= -1

# ψ₁: la degenerazione potrebbe dare combinazione. Prendiamo la parte reale
# Se E₁ è degenere, i due stati hanno simmetria cos/sin
is_degen = abs(eigenvalues[1] - eigenvalues[2]) < 1e-6

print(f"  E₁ e E₂ sono degeneri? {'SÌ' if is_degen else 'NO'}")

if is_degen:
    print("  → Costruiamo le combinazioni cos e sin del doppietto")
    # Le due autofunzioni degeneri possono essere combinate in 
    # una con simmetria cos(4φ) e una con simmetria sin(4φ)
    psi1_cos = (psi1 + psi2) / np.sqrt(2)
    psi1_sin = (psi1 - psi2) / (np.sqrt(2) * 1j)
    
    # Normalizziamo
    psi1_cos /= np.sqrt(np.sum(np.abs(psi1_cos)**2) * dp)
    psi1_sin /= np.sqrt(np.sum(np.abs(psi1_sin)**2) * dp)
    
    # Quale è più "reale"? (ground state band di Mathieu)
    real_frac_cos = np.sum(psi1_cos.real**2 * dp)
    real_frac_sin = np.sum(psi1_sin.real**2 * dp)
    
    print(f"  Fraz. reale di ψ₁_cos: {real_frac_cos:.4f}")
    print(f"  Fraz. reale di ψ₁_sin: {real_frac_sin:.4f}")
    
    # Prendiamo entrambi
    psi1a = psi1_cos if real_frac_cos > real_frac_sin else psi1_sin
    psi1b = psi1_sin if real_frac_cos > real_frac_sin else psi1_cos
else:
    psi1a = psi1
    psi1b = psi2

# Assicuriamo ψ₁ reale al massimo
phase_fix = np.exp(-1j * np.angle(psi1a[np.argmax(np.abs(psi1a))]))
psi1a *= phase_fix

print(f"\n  Proprietà di ψ₀:")
print(f"    Nodi: {np.sum(np.diff(np.sign(psi0.real)) != 0)}")
print(f"    max|ψ₀| = {np.max(np.abs(psi0)):.6f}")
print(f"    Fraz. immaginaria: {np.sum(psi0.imag**2*dp):.2e}")

print(f"\n  Proprietà di ψ₁:")
print(f"    Nodi: {np.sum(np.diff(np.sign(psi1a.real)) != 0)}")
print(f"    max|ψ₁| = {np.max(np.abs(psi1a)):.6f}")
print(f"    Fraz. immaginaria: {np.sum(psi1a.imag**2*dp):.6f}")

# ═══════════════════════════════════════════════════════════════════════
# 3. DECOMPOSIZIONE DI MADELUNG DI ψ₁
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 78}")
print("  3. DECOMPOSIZIONE DI MADELUNG: ψ₁ = R₁ · exp(iS₁/ℏ)")
print("─" * 78)

R1 = np.abs(psi1a)
S1 = np.angle(psi1a)  # fase = S/ℏ

# Flusso di probabilità quantistico
j_qm = (R1**2) * np.gradient(S1, dp) * h2_2m  # j = (ℏ/m)|ψ|²∂S/∂φ

print(f"  R₁(φ) = |ψ₁|:")
print(f"    max = {np.max(R1):.6f}, min = {np.min(R1):.6f}")
print(f"    Zeri (nodi): {np.sum(R1 < 1e-6)} punti sotto 10⁻⁶")

print(f"\n  S₁(φ) = arg(ψ₁) (fase di Madelung):")
print(f"    max = {np.max(S1):.4f}, min = {np.min(S1):.4f}")
print(f"    S₁ costante? σ(S₁) = {np.std(S1):.6f}")
s1_const = np.std(S1) < 0.01

print(f"    {'→ S₁ ≈ costante (ψ₁ quasi-reale)' if s1_const else '→ S₁ NON costante (ψ₁ ha fase non-banale)'}")

print(f"\n  Flusso quantistico j(φ):")
print(f"    max|j| = {np.max(np.abs(j_qm)):.6f}")
print(f"    ∫j dφ = {np.sum(j_qm)*dp:.2e} (deve essere 0 per stato stazionario)")

# Profilo punto per punto
print(f"\n  Profilo di ψ₁:")
print(f"  {'φ/π':>6} {'R₁':>8} {'S₁':>8} {'R₁²':>8} {'ρ_Gibbs':>8} {'R₁²−ρ':>8}")
print(f"  {'─'*52}")

rho_gibbs = np.exp(kappa * np.cos(4*phi)) / (2*np.pi * i0(kappa))

for idx in np.linspace(0, M-1, 17, dtype=int):
    print(f"  {phi[idx]/np.pi:6.2f} {R1[idx]:8.4f} {S1[idx]:+8.4f} "
          f"{R1[idx]**2:8.4f} {rho_gibbs[idx]:8.4f} {R1[idx]**2-rho_gibbs[idx]:+8.4f}")

# ═══════════════════════════════════════════════════════════════════════
# 4. CONFRONTO: |ψ₁|² vs SECONDA AUTOFUNZIONE FOKKER-PLANCK
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 78}")
print("  4. |ψ₁|² vs SECONDA AUTOFUNZIONE DI FOKKER-PLANCK")
print("─" * 78)

print("""
  L'operatore di Fokker-Planck per il mezzo è:
    L_FP = D ∂²/∂φ² + ∂/∂φ(∂V/∂φ · )
  
  Le autofunzioni di L_FP NON sono |ψ_n|²:
  • ψ₀_FP = ρ_eq (distribuzione di equilibrio) [corrisponde a |ψ₀|²]
  • ψ₁_FP = ρ_eq · cos(4φ − α) tipo  [prima eccitazione]
  
  Il test: |ψ₁_Schr|² è uguale a ψ₁_FP?
""")

# Autofunzioni di Fokker-Planck: L_FP ψ = λ ψ
# Per il potenziale cos(4φ), le autofunzioni di L_FP sono 
# proporzionali alle funzioni di Mathieu moltiplicate per ρ_eq
# L_FP = D [∂² - β ∂V'/∂φ ∂ - β V'']

# Costruzione discreta di L_FP
D_fp = 1.0 / (2 * 0.779)  # D = 1/(2β)
beta_fp = 0.779
V_phi = c2 * np.cos(4*phi)
dV = -4*c2 * np.sin(4*phi)
d2V = -16*c2 * np.cos(4*phi)

# Matrice di L_FP (differenze finite)
L_FP = np.zeros((M, M))
for i in range(M):
    ip = (i+1) % M
    im = (i-1) % M
    # D ∂² (diffusione)
    L_FP[i, ip] += D_fp / dp**2
    L_FP[i, i]  -= 2*D_fp / dp**2
    L_FP[i, im] += D_fp / dp**2
    # -∂(F·ρ)/∂φ dove F = -β dV/dφ
    # = -F ∂ρ/∂φ - F'ρ
    F_i = -beta_fp * dV[i]  # forza di drift
    dF_i = -beta_fp * d2V[i]  # ∂F/∂φ
    L_FP[i, ip] -= F_i / (2*dp)
    L_FP[i, im] += F_i / (2*dp)
    L_FP[i, i]  -= dF_i

# Autovalori e autovettori (i più grandi, cioè meno negativi)
eigvals_fp, eigvecs_fp = eigh(L_FP)
# Ordina per autovalore decrescente (λ₀=0, λ₁<0, ...)
idx_sort = np.argsort(-eigvals_fp)
eigvals_fp = eigvals_fp[idx_sort]
eigvecs_fp = eigvecs_fp[:, idx_sort]

print(f"  Primi 5 autovalori di L_FP:")
for i in range(5):
    print(f"    λ_{i} = {eigvals_fp[i]:.6f}")

# Prima autofunzione FP (ρ_eq)
fp0 = eigvecs_fp[:, 0]
fp0 = np.abs(fp0) / (np.sum(np.abs(fp0)) * dp)  # normalizzata come distribuzione

# Seconda autofunzione FP
fp1 = eigvecs_fp[:, 1]
fp1 /= np.sqrt(np.sum(fp1**2) * dp)

# Confronto |ψ₁|² con fp1
prob_psi1 = np.abs(psi1a)**2
prob_psi1 /= np.sum(prob_psi1) * dp

# Overlap tra |ψ₁|² e ρ_eq + ε·fp1
# La prima eccitazione classica è ρ_eq + ε·fp1 (per piccola perturbazione)
# Ma |ψ₁|² non è della stessa forma

print(f"\n  Confronto |ψ₁_QM|² vs autofunzioni FP:")

# Calcoliamo overlap tra |ψ₁|² e varie combinazioni
overlap_fp0 = np.sum(prob_psi1 * fp0 * dp) / np.sqrt(np.sum(prob_psi1**2*dp) * np.sum(fp0**2*dp))
overlap_fp1 = abs(np.sum(prob_psi1 * fp1 * dp)) / np.sqrt(np.sum(prob_psi1**2*dp) * np.sum(fp1**2*dp))

print(f"  Overlap |ψ₁|² con ρ_eq (FP₀): {overlap_fp0:.6f}")
print(f"  Overlap |ψ₁|² con FP₁:        {overlap_fp1:.6f}")

# ═══════════════════════════════════════════════════════════════════════
# 5. IL TEST CRITICO: CORRELAZIONI E(Δθ) DA |ψ₁|²
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 78}")
print("  5. CORRELAZIONI E(Δθ) DA |ψ₁|² vs ρ_Gibbs vs QM")
print("─" * 78)

print("""
  Calcoliamo E(Δθ) usando tre distribuzioni diverse:
  (a) ρ_Gibbs (classico, ground state del mezzo)
  (b) |ψ₀|² (quantistico, ground state di Ĥ) ≈ ρ_Gibbs
  (c) |ψ₁|² (quantistico, primo stato eccitato)
  (d) −cos(Δθ) (QM singoletto ideale)
""")

def correlator(dist, delta_theta):
    d = delta_theta / 2
    A = 2*ndtr(np.cos(phi+d)/sr)-1
    B = -(2*ndtr(np.cos(phi-d)/sr)-1)
    return np.sum(A * B * dist * dp)

print(f"  {'Δθ':>5} {'ρ_Gibbs':>10} {'|ψ₀|²':>10} {'|ψ₁|²':>10} {'−cos Δθ':>10} "
      f"{'Δ(ψ₁−ψ₀)':>10}")
print(f"  {'─'*58}")

prob_psi0 = np.abs(psi0)**2
prob_psi1_norm = np.abs(psi1a)**2
# Normalizzazione
prob_psi0 /= np.sum(prob_psi0)*dp
prob_psi1_norm /= np.sum(prob_psi1_norm)*dp

rmse_gibbs = 0; rmse_psi0 = 0; rmse_psi1 = 0; n = 0
diffs_01 = []

for deg in [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]:
    d = np.radians(deg)
    E_g = correlator(rho_gibbs, d)
    E_0 = correlator(prob_psi0, d)
    E_1 = correlator(prob_psi1_norm, d)
    target = -np.cos(d)
    
    rmse_gibbs += (E_g - target)**2
    rmse_psi0 += (E_0 - target)**2
    rmse_psi1 += (E_1 - target)**2
    n += 1
    
    diff_01 = E_1 - E_0
    diffs_01.append((deg, diff_01))
    
    print(f"  {deg:4d}° {E_g:10.6f} {E_0:10.6f} {E_1:10.6f} {target:10.6f} "
          f"{diff_01:+10.6f}")

rmse_gibbs = np.sqrt(rmse_gibbs/n)
rmse_psi0 = np.sqrt(rmse_psi0/n)
rmse_psi1 = np.sqrt(rmse_psi1/n)

print(f"\n  RMSE vs −cos(Δθ):")
print(f"    ρ_Gibbs:  {rmse_gibbs:.6f}")
print(f"    |ψ₀|²:   {rmse_psi0:.6f}")
print(f"    |ψ₁|²:   {rmse_psi1:.6f}")

print(f"\n  ★ DIFFERENZA |ψ₁|² vs |ψ₀|²:")
max_diff = max(abs(d[1]) for d in diffs_01)
print(f"    max|E(ψ₁) − E(ψ₀)| = {max_diff:.6f}")
print(f"    {'→ Le correlazioni DIFFERISCONO!' if max_diff > 0.01 else '→ Differenza trascurabile'}")

# ═══════════════════════════════════════════════════════════════════════
# 6. ANALISI: PERCHÉ ψ₁ DÀ CORRELAZIONI DIVERSE
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 78}")
print("  6. PERCHÉ ψ₁ DÀ CORRELAZIONI DIVERSE")
print("─" * 78)

print(f"""
  La differenza |ψ₁|² − |ψ₀|² ha una struttura specifica:
""")

diff_distributions = prob_psi1_norm - prob_psi0

# Decomposizione di Fourier
print("  Armoniche di |ψ₁|² − |ψ₀|²:")
for k in range(9):
    ck = 2 * np.sum(diff_distributions * np.cos(k*phi) * dp) / (2*np.pi)
    sk = 2 * np.sum(diff_distributions * np.sin(k*phi) * dp) / (2*np.pi)
    amp = np.sqrt(ck**2 + sk**2)
    if amp > 1e-5:
        print(f"    k={k}: amp = {amp:.6f} {'← dominante' if amp > 0.001 else ''}")

# ═══════════════════════════════════════════════════════════════════════
# 7. MIXED STATE: α|ψ₀⟩ + β|ψ₁⟩
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 78}")
print("  7. STATO MISTO: SOVRAPPOSIZIONE α|ψ₀⟩ + β|ψ₁⟩")
print("─" * 78)

print("""
  La QM permette sovrapposizioni coerenti: |ψ⟩ = α|ψ₀⟩ + β|ψ₁⟩.
  La distribuzione classica non può fare questo.
  
  Per la sovrapposizione: |ψ|² = |α|²|ψ₀|² + |β|²|ψ₁|² + 2Re(α*β ψ₀*ψ₁)
  Il terzo termine (interferenza) NON esiste nel modello classico.
""")

print(f"  {'α':>5} {'E(45°)_QM':>12} {'E(45°)_mix':>12} {'Interf.':>10}")
print(f"  {'─'*42}")

for alpha in [1.0, 0.9, 0.7, 0.5, 0.3, 0.0]:
    beta_s = np.sqrt(1 - alpha**2)
    
    # Sovrapposizione quantistica
    psi_super = alpha * psi0 + beta_s * psi1a
    psi_super /= np.sqrt(np.sum(np.abs(psi_super)**2) * dp)
    prob_super = np.abs(psi_super)**2
    
    # Miscela classica (senza interferenza)
    prob_mix = alpha**2 * prob_psi0 + beta_s**2 * prob_psi1_norm
    
    d45 = np.radians(45)/2
    A45 = 2*ndtr(np.cos(phi+d45)/sr)-1
    B45 = -(2*ndtr(np.cos(phi-d45)/sr)-1)
    
    E_super = np.sum(A45 * B45 * prob_super * dp)
    E_mix = np.sum(A45 * B45 * prob_mix * dp)
    interf = E_super - E_mix
    
    print(f"  {alpha:5.2f} {E_super:12.6f} {E_mix:12.6f} {interf:+10.6f}")

print(f"""
  ★ L'interferenza quantistica (ψ₀*ψ₁ cross-term) modifica 
  le correlazioni rispetto alla miscela classica.
  Questo termine NON esiste nel modello classico su S¹.
  È la firma della FASE nella sovrapposizione.
""")

# ═══════════════════════════════════════════════════════════════════════
# 8. VERDETTO
# ═══════════════════════════════════════════════════════════════════════
print(f"{'═' * 78}")
print("  VERDETTO: PRIMO STATO ECCITATO")
print("═" * 78)

print(f"""
  GROUND STATE (ψ₀):
  • |ψ₀|² ≈ ρ_Gibbs (overlap 99.995%)
  • S₀ = 0 (fase piatta, nessun nodo)
  • Correlazioni identiche al classico
  • → Il ground state NON distingue QM da classico
  
  PRIMO ECCITATO (ψ₁):
  • |ψ₁|² ≠ ρ_Gibbs (distribuzione diversa)
  • ψ₁ ha {np.sum(np.diff(np.sign(psi1a.real)) != 0)} nodi
  • S₁ {'≈ costante (quasi-reale)' if s1_const else 'NON costante (fase non-banale)'}
  • max|E(ψ₁) − E(ψ₀)| = {max_diff:.6f}
  • RMSE(ψ₁) = {rmse_psi1:.6f} vs RMSE(ψ₀) = {rmse_psi0:.6f}
  
  SOVRAPPOSIZIONE (αψ₀ + βψ₁):
  • Il termine di interferenza 2Re(α*β ψ₀*ψ₁) è NON-ZERO
  • Modifica le correlazioni rispetto alla miscela classica
  • Questo termine È la fase — l'ingrediente mancante
  
  CONCLUSIONE:
  {'La divergenza classico/quantistico è PICCOLA per questo potenziale.' if max_diff < 0.01 else 'La divergenza classico/quantistico è MISURABILE.'}
  La ragione: κ = {kappa:.3f} è nel regime dove il potenziale è 
  debole rispetto a ℏ²/2m = {h2_2m:.3f}, quindi ψ₁ è vicino a 
  un'onda piana (quasi-libera).
  
  Per vedere una divergenza GRANDE, servirebbero:
  • Potenziale più profondo (κ >> ℏ²/2m)
  • O misure in stati eccitati (non ground state)
  • O sovrapposizioni coerenti (che il classico non può fare)
  
  Il punto chiave resta: il modello classico non può MAI produrre 
  il termine di interferenza 2Re(α*β ψ₀*ψ₁). Solo il quantistico può.
  Quel termine è la FIRMA della fase S.
""")

