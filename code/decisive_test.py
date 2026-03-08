"""
═══════════════════════════════════════════════════════════════════════════
  IL TEST DECISIVO
  
  1. Costruire ψ = √ρ · exp(iS/ℏ)
  2. Estrarre S dal flusso di probabilità della Langevin
  3. Verificare se ψ soddisfa una Schrödinger efficace
  4. Misurare il residuo
═══════════════════════════════════════════════════════════════════════════
"""
import numpy as np
from scipy.special import i0, i1

M = 2048
phi = np.linspace(0, 2*np.pi, M, endpoint=False)
dp = 2*np.pi / M
c2 = -0.876; beta = 0.779

print("=" * 78)
print("  IL TEST DECISIVO: ψ = √ρ · exp(iS/ℏ)")
print("=" * 78)

# ═══════════════════════════════════════════════════════════════════════
# PASSO 1: LA DISTRIBUZIONE ρ(φ) DEL MEZZO
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 78}")
print("  PASSO 1: ρ(φ) dalla distribuzione stazionaria")
print("─" * 78)

kappa = beta * abs(c2)
rho = np.exp(kappa * np.cos(4*phi)) / (2*np.pi * i0(kappa))

# Verifica normalizzazione
norm_rho = np.sum(rho) * dp
print(f"  ∫ρ dφ = {norm_rho:.10f} (deve essere 1)")
print(f"  κ = β|c₂| = {kappa:.6f}")
print(f"  max(ρ) = {np.max(rho):.6f}, min(ρ) = {np.min(rho):.6f}")

# ═══════════════════════════════════════════════════════════════════════
# PASSO 2: √ρ — LA PARTE REALE DI ψ
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 78}")
print("  PASSO 2: R(φ) = √ρ(φ)")
print("─" * 78)

R = np.sqrt(rho)
norm_R2 = np.sum(R**2) * dp
print(f"  ∫R² dφ = {norm_R2:.10f} (deve essere 1)")

# ═══════════════════════════════════════════════════════════════════════
# PASSO 3: ESTRARRE S(φ) DAL FLUSSO DI PROBABILITÀ
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 78}")
print("  PASSO 3: Estrarre S(φ) dal flusso di Smoluchowski")
print("─" * 78)

print("""
  L'equazione di Smoluchowski stazionaria è:
  
    J = −D ∂ρ/∂φ − ρ ∂V/∂φ = costante
  
  In equilibrio su S¹, J = 0 (nessun flusso netto su cerchio).
  Questo significa: il flusso stazionario è NULLO.
  
  Ma nella decomposizione di Madelung:
    ψ = R · exp(iS/ℏ)
  il flusso quantistico è:
    j_QM = (ℏ/m) R² ∂S/∂φ = (ℏ/m) ρ ∂S/∂φ
  
  Se j_QM = 0 (stato stazionario), allora ∂S/∂φ = 0 → S = costante.
  
  STATO STAZIONARIO → S = 0 (a meno di una costante globale).
  La fase è PIATTA per lo stato fondamentale.
  
  Questo è COERENTE con la QM: il ground state ha S = 0.
  Solo gli stati eccitati hanno S ≠ 0 (nodi, correnti).
""")

# Per lo stato fondamentale: S = 0, quindi ψ = R = √ρ (REALE)
S_ground = np.zeros(M)
psi_ground = R * np.exp(1j * S_ground)  # = R (reale)

print(f"  Per il ground state: S(φ) = 0")
print(f"  ψ₀(φ) = √ρ(φ) (puramente reale)")
print(f"  ∫|ψ₀|² dφ = {np.sum(np.abs(psi_ground)**2)*dp:.10f}")

# ═══════════════════════════════════════════════════════════════════════
# PASSO 4: VERIFICARE SCHRÖDINGER  Ĥψ = Eψ
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 78}")
print("  PASSO 4: Verifica Ĥψ₀ = E₀ψ₀")
print("─" * 78)

print("""
  L'equazione di Schrödinger stazionaria su S¹ è:
  
    −(ℏ²/2m) ∂²ψ/∂φ² + V(φ)ψ = Eψ
  
  Con ψ₀ = √ρ, possiamo:
  (a) Scegliere ℏ²/2m come parametro libero
  (b) Calcolare V_eff tale che Ĥψ₀ = E₀ψ₀
  (c) Confrontare V_eff con V_mezzo = c₂cos(4φ)
  
  Il potenziale effettivo si ottiene dalla formula inversa:
  
    V_eff(φ) = E₀ + (ℏ²/2m) · (∂²ψ₀/∂φ²) / ψ₀
""")

# Calcoliamo ∂²R/∂φ² numericamente
dR = np.gradient(R, dp)
d2R = np.gradient(dR, dp)

# Il rapporto (∂²R/∂φ²)/R è il potenziale quantistico di Bohm
# V_eff = E - (ℏ²/2m)(d²R/dφ²)/R

Q_ratio = d2R / (R + 1e-30)  # evita divisione per zero

print("  Scansione di ℏ²/(2m) per minimizzare il residuo Ĥψ - Eψ:")
print(f"  {'ℏ²/2m':>8} {'E₀':>10} {'RMSE(V_eff, V_mezzo)':>22} {'max|V_eff−V|':>14}")
print(f"  {'─'*58}")

best_rmse = float('inf')
best_hbar = 0
best_E0 = 0

for h2_2m in np.arange(0.001, 0.5, 0.001):
    # V_eff = E₀ + h2_2m * d²R/(R·dφ²)
    # Scegliamo E₀ tale che ⟨V_eff⟩ = ⟨V_mezzo⟩
    V_eff_raw = h2_2m * Q_ratio
    
    # V_mezzo originale (nel quadro di Schrödinger, V è quello dell'azione)
    # Il potenziale del mezzo nella Smoluchowski è: V_smol = -c₂ cos(4φ) / (da S = -V)
    # Nella Schrödinger, il potenziale è libero. Confrontiamo con cos(4φ) riscalato.
    V_mezzo = c2 * np.cos(4*phi)  # il potenziale dell'azione
    
    # Aggiustiamo E₀ per minimizzare la distanza
    E0 = np.sum((V_mezzo - V_eff_raw) * rho * dp)  # media pesata
    V_eff = V_eff_raw + E0
    
    # RMSE pesato con ρ (dove ψ è grande conta di più)
    rmse = np.sqrt(np.sum((V_eff - V_mezzo)**2 * rho * dp))
    maxdev = np.max(np.abs((V_eff - V_mezzo) * R))  # pesato con R
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_hbar = h2_2m
        best_E0 = E0
    
    if abs(h2_2m - 0.01) < 0.001 or abs(h2_2m - 0.05) < 0.001 or \
       abs(h2_2m - 0.1) < 0.001 or abs(h2_2m - 0.02) < 0.001 or \
       abs(h2_2m - 0.03) < 0.001 or abs(h2_2m - 0.005) < 0.001:
        print(f"  {h2_2m:8.3f} {E0:10.4f} {rmse:22.6f} {maxdev:14.6f}")

print(f"\n  ★ OTTIMO: ℏ²/(2m) = {best_hbar:.4f}, RMSE = {best_rmse:.6f}")

# Calcolo dettagliato con l'ottimo
h2_2m = best_hbar
V_eff_opt = h2_2m * Q_ratio + best_E0
V_mezzo = c2 * np.cos(4*phi)

# Il residuo di Schrödinger: r(φ) = [−h²∂²ψ/∂φ² + Vψ − Eψ] / ψ
# Con V = V_mezzo, cerchiamo quanto V_eff ≠ V_mezzo

print(f"\n  Confronto punto per punto V_eff vs V_mezzo (all'ottimo):")
print(f"  {'φ/π':>6} {'V_mezzo':>10} {'V_eff':>10} {'V_eff−V':>10} {'ρ(φ)':>8}")
print(f"  {'─'*48}")

for idx in np.linspace(0, M-1, 13, dtype=int):
    print(f"  {phi[idx]/np.pi:6.2f} {V_mezzo[idx]:10.4f} {V_eff_opt[idx]:10.4f} "
          f"{V_eff_opt[idx]-V_mezzo[idx]:+10.4f} {rho[idx]:8.4f}")

# ═══════════════════════════════════════════════════════════════════════
# PASSO 5: TEST DIRETTO — Ĥψ₀ = E₀ψ₀ CON V_MEZZO
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 78}")
print("  PASSO 5: Residuo diretto di Schrödinger")
print("─" * 78)

print("""
  Test più onesto: prendiamo V = c₂cos(4φ) come dato, calcoliamo 
  Ĥψ₀ = −(ℏ²/2m)ψ₀'' + V·ψ₀, e misuriamo quanto Ĥψ₀ è 
  proporzionale a ψ₀ (cioè quanto è vicino a un autostato).
  
  Se Ĥψ₀ = E₀ψ₀ esattamente, il rapporto Ĥψ₀/ψ₀ = costante.
""")

for h2_2m in [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, best_hbar]:
    Hpsi = -h2_2m * d2R + V_mezzo * R
    
    # Se Hψ = Eψ, allora Hψ/ψ = costante
    ratio = Hpsi / (R + 1e-30)
    
    # Media e deviazione del rapporto (pesato con R²)
    E_mean = np.sum(ratio * R**2 * dp) / np.sum(R**2 * dp)
    E_std = np.sqrt(np.sum((ratio - E_mean)**2 * R**2 * dp) / np.sum(R**2 * dp))
    
    # Residuo relativo
    res_rel = E_std / abs(E_mean) if abs(E_mean) > 1e-10 else float('inf')
    
    # Residuo L² normalizzato: ||Ĥψ − Eψ|| / ||ψ||
    residual_vec = Hpsi - E_mean * R
    res_L2 = np.sqrt(np.sum(residual_vec**2 * dp)) / np.sqrt(np.sum(R**2 * dp))
    
    marker = " ← OTTIMO" if abs(h2_2m - best_hbar) < 0.001 else ""
    print(f"  ℏ²/2m={h2_2m:.4f}: E₀={E_mean:+.6f}, σ(E)={E_std:.6f}, "
          f"σ/|E|={res_rel:.4f}, ||res||/||ψ||={res_L2:.6f}{marker}")

# ═══════════════════════════════════════════════════════════════════════
# PASSO 6: CONFRONTO CON IL VERO GROUND STATE DI Ĥ
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 78}")
print("  PASSO 6: Ground state esatto di Ĥ vs √ρ del mezzo")
print("─" * 78)

print("""
  Costruiamo la matrice Hamiltoniana H nella base di Fourier 
  e troviamo il vero ground state ψ₀_QM. Poi confrontiamo con √ρ.
""")

# Base di Fourier: exp(inφ)/√(2π) per n = -N,...,N
N_basis = 30
n_range = np.arange(-N_basis, N_basis+1)
dim = len(n_range)

# Matrice Hamiltoniana in base di Fourier
# T_nn' = (ℏ²/2m) n² δ_nn'  (energia cinetica)
# V_nn' = c₂/2 [δ_{n,n'+4} + δ_{n,n'-4}]  (potenziale cos(4φ))

for h2_2m in [best_hbar, 0.01, 0.02, 0.05]:
    H_mat = np.zeros((dim, dim))
    
    # Cinetica
    for i, n in enumerate(n_range):
        H_mat[i, i] = h2_2m * n**2
    
    # Potenziale c₂ cos(4φ) = c₂/2 [e^(4iφ) + e^(-4iφ)]
    for i, n in enumerate(n_range):
        for j, m in enumerate(n_range):
            if n - m == 4 or n - m == -4:
                H_mat[i, j] += c2 / 2
    
    # Diagonalizzazione
    eigenvalues, eigenvectors = np.linalg.eigh(H_mat)
    
    E0_exact = eigenvalues[0]
    psi0_coeffs = eigenvectors[:, 0]
    
    # Ricostruire ψ₀(φ) dalla base di Fourier
    psi0_exact = np.zeros(M, dtype=complex)
    for i, n in enumerate(n_range):
        psi0_exact += psi0_coeffs[i] * np.exp(1j * n * phi) / np.sqrt(2*np.pi)
    
    # Normalizzazione
    norm_exact = np.sqrt(np.sum(np.abs(psi0_exact)**2) * dp)
    psi0_exact /= norm_exact
    
    # Assicuriamo fase globale positiva
    if np.sum(psi0_exact.real) < 0:
        psi0_exact *= -1
    
    # Confronto con √ρ normalizzato
    R_norm = R / np.sqrt(np.sum(R**2) * dp)
    
    # Overlap ⟨ψ₀_QM | √ρ⟩
    overlap = abs(np.sum(psi0_exact.real * R_norm * dp))
    
    # RMSE
    rmse = np.sqrt(np.sum((np.abs(psi0_exact) - R_norm)**2 * dp))
    
    # ψ₀ è reale per ground state? (V è pari)
    imag_frac = np.sum(psi0_exact.imag**2 * dp) / np.sum(np.abs(psi0_exact)**2 * dp)
    
    print(f"  ℏ²/2m = {h2_2m:.4f}:")
    print(f"    E₀_esatto = {E0_exact:.6f}")
    print(f"    Gap E₁−E₀ = {eigenvalues[1]-eigenvalues[0]:.6f}")
    print(f"    |⟨ψ₀|√ρ⟩| = {overlap:.8f}")
    print(f"    RMSE(|ψ₀|, √ρ) = {rmse:.8f}")
    print(f"    Fraz. immaginaria = {imag_frac:.2e}")
    
    if overlap > 0.999:
        print(f"    ★ MATCH ECCELLENTE: √ρ ≈ ψ₀ con overlap {overlap:.6f}")
    elif overlap > 0.99:
        print(f"    ★ BUON MATCH: overlap {overlap:.6f}")
    elif overlap > 0.9:
        print(f"    → MATCH PARZIALE: overlap {overlap:.6f}")
    else:
        print(f"    → MATCH SCARSO: overlap {overlap:.6f}")
    print()

# ═══════════════════════════════════════════════════════════════════════
# PASSO 7: SCANSIONE FINE PER TROVARE ℏ OTTIMALE
# ═══════════════════════════════════════════════════════════════════════
print(f"{'─' * 78}")
print("  PASSO 7: ℏ²/(2m) ottimale per massimizzare overlap")
print("─" * 78)

best_overlap = 0
best_h = 0

for h2_2m in np.arange(0.001, 0.3, 0.001):
    H_mat = np.zeros((dim, dim))
    for i, n in enumerate(n_range):
        H_mat[i, i] = h2_2m * n**2
    for i, n in enumerate(n_range):
        for j, m in enumerate(n_range):
            if abs(n - m) == 4:
                H_mat[i, j] += c2 / 2
    
    eigenvalues, eigenvectors = np.linalg.eigh(H_mat)
    psi0_coeffs = eigenvectors[:, 0]
    
    psi0_exact = np.zeros(M, dtype=complex)
    for i, n in enumerate(n_range):
        psi0_exact += psi0_coeffs[i] * np.exp(1j * n * phi) / np.sqrt(2*np.pi)
    
    norm_exact = np.sqrt(np.sum(np.abs(psi0_exact)**2) * dp)
    psi0_exact /= norm_exact
    if np.sum(psi0_exact.real) < 0:
        psi0_exact *= -1
    
    overlap = abs(np.sum(psi0_exact.real * R_norm * dp))
    
    if overlap > best_overlap:
        best_overlap = overlap
        best_h = h2_2m

print(f"  ★ OTTIMO: ℏ²/(2m) = {best_h:.4f}")
print(f"    Overlap |⟨ψ₀|√ρ⟩| = {best_overlap:.8f}")

# Risultato dettagliato all'ottimo
h2_2m = best_h
H_mat = np.zeros((dim, dim))
for i, n in enumerate(n_range):
    H_mat[i, i] = h2_2m * n**2
for i, n in enumerate(n_range):
    for j, m in enumerate(n_range):
        if abs(n - m) == 4:
            H_mat[i, j] += c2 / 2

eigenvalues, eigenvectors = np.linalg.eigh(H_mat)
psi0_coeffs = eigenvectors[:, 0]

psi0_exact = np.zeros(M, dtype=complex)
for i, n in enumerate(n_range):
    psi0_exact += psi0_coeffs[i] * np.exp(1j * n * phi) / np.sqrt(2*np.pi)
norm_exact = np.sqrt(np.sum(np.abs(psi0_exact)**2) * dp)
psi0_exact /= norm_exact
if np.sum(psi0_exact.real) < 0:
    psi0_exact *= -1

rmse_final = np.sqrt(np.sum((np.abs(psi0_exact) - R_norm)**2 * dp))

print(f"    E₀ = {eigenvalues[0]:.6f}, E₁ = {eigenvalues[1]:.6f}")
print(f"    Gap = {eigenvalues[1]-eigenvalues[0]:.6f}")
print(f"    RMSE(|ψ₀|, √ρ) = {rmse_final:.8f}")

# Relazione β ↔ ℏ
# Gibbs: ρ ∝ exp(βc₂cos(4φ)) = exp(κcos(4φ)) con κ = β|c₂|
# Schrödinger con V = c₂cos(4φ) e ℏ²/2m:
# Il parametro adimensionale è κ/h = β|c₂|/(ℏ²/2m)
ratio_kh = kappa / best_h

print(f"\n  RELAZIONE β ↔ ℏ:")
print(f"    κ = β|c₂| = {kappa:.4f}")
print(f"    ℏ²/(2m) ottimale = {best_h:.4f}")
print(f"    κ / [ℏ²/(2m)] = {ratio_kh:.4f}")

# ═══════════════════════════════════════════════════════════════════════
# PASSO 8: IL CORRELATORE E(Δθ) DAL GROUND STATE QUANTISTICO
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 78}")
print("  PASSO 8: Correlatore E(Δθ) dal ψ₀ quantistico vs classico")
print("─" * 78)

from scipy.special import ndtr
sr = 0.005

print(f"  ℏ²/(2m) = {best_h:.4f}")
print(f"\n  {'Δθ':>5} {'E(classico)':>12} {'E(quantist.)':>14} {'−cos(Δθ)':>10} "
      f"{'err_cl':>8} {'err_qm':>8}")
print(f"  {'─'*60}")

rmse_cl = 0; rmse_qm = 0; n = 0

for deg in [0, 15, 30, 45, 60, 75, 90, 120, 135, 150, 165, 180]:
    d = np.radians(deg) / 2
    A = 2*ndtr(np.cos(phi+d)/sr)-1
    B = -(2*ndtr(np.cos(phi-d)/sr)-1)
    
    # Classico: ∫ A·B·ρ dφ
    E_cl = np.sum(A * B * rho * dp)
    
    # Quantistico: ⟨ψ₀|A·B|ψ₀⟩ = ∫ |ψ₀|² A·B dφ
    psi0_prob = np.abs(psi0_exact)**2
    E_qm = np.sum(A * B * psi0_prob * dp)
    
    target = -np.cos(np.radians(deg))
    
    rmse_cl += (E_cl - target)**2
    rmse_qm += (E_qm - target)**2
    n += 1
    
    print(f"  {deg:4d}° {E_cl:12.6f} {E_qm:14.6f} {target:10.6f} "
          f"{abs(E_cl-target):8.4f} {abs(E_qm-target):8.4f}")

rmse_cl = np.sqrt(rmse_cl/n)
rmse_qm = np.sqrt(rmse_qm/n)

print(f"\n  RMSE classico vs −cos(Δθ): {rmse_cl:.6f}")
print(f"  RMSE quantistico vs −cos(Δθ): {rmse_qm:.6f}")
print(f"  Rapporto: {rmse_cl/rmse_qm:.2f}x")

# ═══════════════════════════════════════════════════════════════════════
# VERDETTO
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 78}")
print("  VERDETTO DEL TEST DECISIVO")
print("═" * 78)

print(f"""
  DATI:
  • ρ(φ) = distribuzione di Gibbs del mezzo con κ = {kappa:.4f}
  • ψ₀(φ) = ground state di Ĥ = −(ℏ²/2m)∂² + c₂cos(4φ)
  • ℏ²/(2m) ottimale = {best_h:.4f}
  
  RISULTATI:
  1) L'overlap |⟨ψ₀|√ρ⟩| = {best_overlap:.6f}
     {'→ √ρ È il ground state di Schrödinger (a ℏ fissato)' if best_overlap > 0.99 else '→ √ρ NON è esattamente il ground state'}
  
  2) RMSE(|ψ₀|, √ρ) = {rmse_final:.6f}
     {'→ Differenza trascurabile' if rmse_final < 0.01 else '→ Differenza significativa: ' + str(rmse_final)}
  
  3) Correlatore: 
     RMSE classico (ρ): {rmse_cl:.6f}
     RMSE quantistico (|ψ₀|²): {rmse_qm:.6f}
     {'→ Il quantistico è MIGLIORE' if rmse_qm < rmse_cl else '→ Il classico è migliore o uguale'}
  
  4) Relazione κ/(ℏ²/2m) = {ratio_kh:.2f}
     Questo rapporto fissa la relazione tra la "temperatura" del 
     mezzo (β) e la costante di Planck (ℏ).

  INTERPRETAZIONE:
  {'La distribuzione di Gibbs del mezzo ρ ∝ exp(κcos4φ) COINCIDE con |ψ₀|² del ground state di Schrödinger con potenziale cos(4φ), per un valore specifico di ℏ.' if best_overlap > 0.99 else 'La corrispondenza √ρ ↔ ψ₀ è approssimata, non esatta. Il gap indica che Gibbs e Born non sono identici in questo regime.'}
""")

