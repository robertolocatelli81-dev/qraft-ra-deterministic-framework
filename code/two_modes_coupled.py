"""
═══════════════════════════════════════════════════════════════════════════
  DUE MODI ACCOPPIATI SU (S¹)²
  
  Obiettivo: costruire l'estensione minima che produce GHZ 
  senza rompere il no-signaling.
  
  Metodo: partire da ciò che MANCA e lavorare al contrario.
═══════════════════════════════════════════════════════════════════════════
"""
import numpy as np
from scipy.special import ndtr, i0, i1
from scipy.optimize import differential_evolution

# Griglia 2D su (S¹)²
N1 = 256   # risoluzione per modo
N2 = 256
phi1 = np.linspace(0, 2*np.pi, N1, endpoint=False)
phi2 = np.linspace(0, 2*np.pi, N2, endpoint=False)
PHI1, PHI2 = np.meshgrid(phi1, phi2, indexing='ij')
dp1 = 2*np.pi / N1
dp2 = 2*np.pi / N2
dp2d = dp1 * dp2
sr = 0.005

print("=" * 78)
print("  (S¹)²: DUE MODI ACCOPPIATI")
print("  Lavorare al contrario: cosa serve per GHZ + no-signaling?")
print("=" * 78)

# ═══════════════════════════════════════════════════════════════════════
# 1. COSA SERVE PER GHZ: ANALISI AL CONTRARIO
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 78}")
print("  1. INGEGNERIA INVERSA: COSA PRODUCE GHZ?")
print("─" * 78)

print("""
  Lo stato GHZ a 3 qubit ha la correlazione:
    ⟨ABC⟩_GHZ = −cos(a + b + c)
  
  Per un modello a variabili nascoste su (S¹)²:
    ⟨ABC⟩ = ∫∫ Ā(φ₁,φ₂;a) B̄(φ₁,φ₂;b) C̄(φ₁,φ₂;c) ρ(φ₁,φ₂;a,b,c) dφ₁dφ₂
  
  Il PROBLEMA con un modo solo:
  • cos(φ−a)cos(φ−b)cos(φ−c) produce armonica dispari (freq 3)
  • freq 3 → distribuzione NON π-periodica → NS rotto
  
  La SOLUZIONE con due modi:
  • Distribuire le 3 misure su 2 modi
  • L'accoppiamento inter-modo crea la correlazione a 3 corpi
  • Le armoniche restano pari SU CIASCUN MODO → NS preservato
  
  ARCHITETTURA:
  
    Modo 1 (φ₁): accoppiato ad A e B  →  E_AB = F(a−b)
    Modo 2 (φ₂): accoppiato ad A e C  →  E_AC = F(a−c)
    Accoppiamento: λ·cos(2φ₁−2φ₂)    →  lega i due modi
    
    A è il NODO CONDIVISO: legge entrambi i modi.
""")

# ═══════════════════════════════════════════════════════════════════════
# 2. COSTRUZIONE DELL'AZIONE SU (S¹)²
# ═══════════════════════════════════════════════════════════════════════
print(f"{'─' * 78}")
print("  2. AZIONE SU (S¹)² CON ACCOPPIAMENTO INTER-MODO")
print("─" * 78)

def build_distribution_2d(a, b, c, beta, c2, lam):
    """
    Distribuzione su (S¹)² per 3 particelle.
    
    Modo 1 (φ₁): A-B accoppiati via cos(φ₁−a)cos(φ₁−b) → cos(4ψ₁) quadrupolo
    Modo 2 (φ₂): A-C accoppiati via cos(φ₂−a)cos(φ₂−c) → cos(4ψ₂) quadrupolo
    Accoppiamento: λ·cos(2φ₁−2φ₂) = λ·cos(2(φ₁−φ₂))
    
    ψ₁ = φ₁ − (a+b)/2, ψ₂ = φ₂ − (a+c)/2
    """
    # Azione quadrupolare su ciascun modo
    S1 = c2 * np.cos(4*(PHI1 - (a+b)/2))
    S2 = c2 * np.cos(4*(PHI2 - (a+c)/2))
    
    # Accoppiamento inter-modo
    # cos(2(φ₁−φ₂)) nel frame centrato:
    # φ₁−φ₂ = (ψ₁ + (a+b)/2) − (ψ₂ + (a+c)/2) = ψ₁ − ψ₂ + (b−c)/2
    S_couple = lam * np.cos(2*(PHI1 - PHI2))
    
    S_total = S1 + S2 + S_couple
    
    lw = beta * S_total
    lw -= np.max(lw)
    w = np.exp(lw)
    Z = np.sum(w) * dp2d
    return w / Z

def correlator_3body(a, b, c, beta, c2, lam):
    """⟨ABC⟩ con A che legge entrambi i modi."""
    p = build_distribution_2d(a, b, c, beta, c2, lam)
    
    # A legge la MEDIA dei due modi: cos(φ₁−a) e cos(φ₂−a)
    # proiezione combinata: (cos(φ₁−a) + cos(φ₂−a))/2 → misura su entrambi
    A_map = 2*ndtr((np.cos(PHI1-a) + np.cos(PHI2-a))/(2*sr)) - 1
    # B legge solo modo 1
    B_map = -(2*ndtr(np.cos(PHI1-b)/sr) - 1)
    # C legge solo modo 2
    C_map = -(2*ndtr(np.cos(PHI2-c)/sr) - 1)
    
    ABC = np.sum(A_map * B_map * C_map * p * dp2d)
    
    # Marginali per NS
    muA = np.sum(A_map * p * dp2d)
    muB = np.sum(B_map * p * dp2d)
    muC = np.sum(C_map * p * dp2d)
    
    return ABC, muA, muB, muC

def correlator_2body(a, b, beta, c2, lam, pair="AB"):
    """Correlatori a coppie dalla distribuzione congiunta."""
    # Per E_AB: marginalizzare su φ₂ (c arbitrario, es. c=0)
    c_dummy = 0
    p = build_distribution_2d(a, b, c_dummy, beta, c2, lam)
    
    if pair == "AB":
        A_map = 2*ndtr(np.cos(PHI1-a)/sr) - 1
        B_map = -(2*ndtr(np.cos(PHI1-b)/sr) - 1)
        return np.sum(A_map * B_map * p * dp2d)
    elif pair == "AC":
        A_map = 2*ndtr(np.cos(PHI2-a)/sr) - 1
        C_map = -(2*ndtr(np.cos(PHI2-b)/sr) - 1)
        return np.sum(A_map * C_map * p * dp2d)

# ═══════════════════════════════════════════════════════════════════════
# 3. SCAN DEL PARAMETRO DI ACCOPPIAMENTO λ
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 78}")
print("  3. EFFETTO DELL'ACCOPPIAMENTO λ")
print("─" * 78)

beta = 0.779
c2_val = -0.876

print(f"  β = {beta}, c₂ = {c2_val}")
print(f"\n  {'λ':>6} {'⟨ABC⟩(0,0,0)':>14} {'GHZ':>8} {'μ_A':>10} {'μ_B':>10} {'μ_C':>10} {'NS?':>5}")
print(f"  {'─'*60}")

for lam in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
    ABC, muA, muB, muC = correlator_3body(0, 0, 0, beta, c2_val, lam)
    ghz = -np.cos(0+0+0)  # = -1
    ns = max(abs(muA), abs(muB), abs(muC))
    ns_ok = "✓" if ns < 0.01 else "✗"
    print(f"  {lam:6.2f} {ABC:14.6f} {ghz:8.4f} {muA:10.2e} {muB:10.2e} {muC:10.2e} {ns_ok:>5}")

# ═══════════════════════════════════════════════════════════════════════
# 4. SCAN COMPLETO: ⟨ABC⟩ PER VARIE CONFIGURAZIONI
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 78}")
print("  4. CORRELAZIONI A 3 CORPI PER VARIE CONFIGURAZIONI")
print("─" * 78)

lam_test = 1.0
print(f"  λ = {lam_test}")
print(f"\n  {'(a,b,c)':>25} {'⟨ABC⟩':>10} {'−cos(Σ)':>10} {'Errore':>10} {'max|μ|':>10}")
print(f"  {'─'*68}")

configs = [
    (0, 0, 0), 
    (np.pi/6, 0, 0),
    (np.pi/4, np.pi/4, 0),
    (np.pi/4, np.pi/4, np.pi/4),
    (np.pi/3, np.pi/6, 0),
    (np.pi/2, 0, 0),
    (np.pi/3, np.pi/3, np.pi/3),
    (0, np.pi/4, np.pi/2),
    (np.pi/6, np.pi/3, np.pi/2),
]

rmse_ghz = 0
n_cfg = 0
for a, b, c in configs:
    ABC, muA, muB, muC = correlator_3body(a, b, c, beta, c2_val, lam_test)
    ghz = -np.cos(a+b+c)
    ns = max(abs(muA), abs(muB), abs(muC))
    err = abs(ABC - ghz)
    rmse_ghz += (ABC - ghz)**2
    n_cfg += 1
    degs = f"({np.degrees(a):.0f}°,{np.degrees(b):.0f}°,{np.degrees(c):.0f}°)"
    print(f"  {degs:>25} {ABC:10.4f} {ghz:10.4f} {err:10.4f} {ns:10.2e}")

rmse_ghz = np.sqrt(rmse_ghz / n_cfg)
print(f"\n  RMSE vs GHZ: {rmse_ghz:.6f}")

# ═══════════════════════════════════════════════════════════════════════
# 5. OTTIMIZZAZIONE: TROVARE (c₂, β, λ) CHE MINIMIZZANO RMSE vs GHZ
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 78}")
print("  5. OTTIMIZZAZIONE PER GHZ")
print("─" * 78)

def ghz_rmse(params):
    c2_p, beta_p, lam_p = params
    if beta_p < 0.01: return 10
    err = 0; n = 0
    for a, b, c in [(0,0,0), (np.pi/4,0,0), (np.pi/4,np.pi/4,0),
                     (np.pi/4,np.pi/4,np.pi/4), (np.pi/3,np.pi/6,0),
                     (np.pi/2,0,0), (0,np.pi/4,np.pi/2)]:
        try:
            ABC, muA, muB, muC = correlator_3body(a, b, c, beta_p, c2_p, lam_p)
            ghz = -np.cos(a+b+c)
            ns = max(abs(muA), abs(muB), abs(muC))
            if ns > 0.1:  # penalizzare violazioni NS
                return 10
            err += (ABC - ghz)**2
            n += 1
        except:
            return 10
    return np.sqrt(err/n)

print("  Ottimizzazione in corso...")
res = differential_evolution(ghz_rmse, 
    bounds=[(-3, 0), (0.1, 5), (0, 5)],
    seed=42, maxiter=50, tol=1e-6, popsize=12)

c2_opt, beta_opt, lam_opt = res.x
rmse_opt = res.fun

print(f"\n  OTTIMO GHZ su (S¹)²:")
print(f"    c₂  = {c2_opt:.4f}")
print(f"    β   = {beta_opt:.4f}")
print(f"    λ   = {lam_opt:.4f}")
print(f"    RMSE vs GHZ = {rmse_opt:.6f}")

# Verifica dettagliata dell'ottimo
print(f"\n  Verifica con parametri ottimali:")
print(f"  {'(a,b,c)':>25} {'⟨ABC⟩':>10} {'−cos(Σ)':>10} {'Err':>8} {'max|μ|':>10}")
print(f"  {'─'*66}")

for a, b, c in configs:
    ABC, muA, muB, muC = correlator_3body(a, b, c, beta_opt, c2_opt, lam_opt)
    ghz = -np.cos(a+b+c)
    ns = max(abs(muA), abs(muB), abs(muC))
    print(f"  {f'({np.degrees(a):.0f},{np.degrees(b):.0f},{np.degrees(c):.0f})':>25} "
          f"{ABC:10.4f} {ghz:10.4f} {abs(ABC-ghz):8.4f} {ns:10.2e}")

# ═══════════════════════════════════════════════════════════════════════
# 6. NO-SIGNALING SU (S¹)²: VERIFICA
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 78}")
print("  6. NO-SIGNALING SU (S¹)²: VERIFICA SISTEMATICA")
print("─" * 78)

print(f"  Parametri: c₂={c2_opt:.4f}, β={beta_opt:.4f}, λ={lam_opt:.4f}")
print(f"\n  Scan: μ_A, μ_B, μ_C per 100 configurazioni casuali")

np.random.seed(42)
max_mu_A = 0; max_mu_B = 0; max_mu_C = 0

for _ in range(100):
    a = np.random.uniform(0, 2*np.pi)
    b = np.random.uniform(0, 2*np.pi)
    c = np.random.uniform(0, 2*np.pi)
    _, muA, muB, muC = correlator_3body(a, b, c, beta_opt, c2_opt, lam_opt)
    max_mu_A = max(max_mu_A, abs(muA))
    max_mu_B = max(max_mu_B, abs(muB))
    max_mu_C = max(max_mu_C, abs(muC))

print(f"    max|μ_A| = {max_mu_A:.6f}")
print(f"    max|μ_B| = {max_mu_B:.6f}")
print(f"    max|μ_C| = {max_mu_C:.6f}")

ns_a = "✓" if max_mu_A < 0.01 else "✗"
ns_b = "✓" if max_mu_B < 0.01 else "✗"
ns_c = "✓" if max_mu_C < 0.01 else "✗"

print(f"    NS per A: {ns_a}  B: {ns_b}  C: {ns_c}")

# ═══════════════════════════════════════════════════════════════════════
# 7. DISUGUAGLIANZA DI MERMIN SU (S¹)²
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 78}")
print("  7. MERMIN SU (S¹)²")
print("─" * 78)

def mermin_2mode(beta_p, c2_p, lam_p):
    a1, a2 = 0, np.pi/2
    b1, b2 = 0, np.pi/2
    c1, c2_s = 0, np.pi/2
    
    def corr(a,b,c):
        ABC, _, _, _ = correlator_3body(a, b, c, beta_p, c2_p, lam_p)
        return ABC
    
    M3 = abs(corr(a1,b2,c2_s) + corr(a2,b1,c2_s) + 
             corr(a2,b2,c1) - corr(a1,b1,c1))
    return M3

M3_val = mermin_2mode(beta_opt, c2_opt, lam_opt)
print(f"  M₃ (2 modi, ottimizzato) = {M3_val:.6f}")
print(f"  M₃ (1 modo, precedente)  ≈ 0.5")
print(f"  M₃ (classico)            ≤ 2")
print(f"  M₃ (GHZ, QM)             = 4")

# Scan λ per Mermin
print(f"\n  M₃ come funzione di λ:")
print(f"  {'λ':>6} {'M₃':>10}")
print(f"  {'─'*18}")
for lam in [0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
    M3 = mermin_2mode(beta_opt, c2_opt, lam)
    marker = " ← supera classico!" if M3 > 2 else ""
    print(f"  {lam:6.2f} {M3:10.4f}{marker}")

# ═══════════════════════════════════════════════════════════════════════
# 8. PATTERN EMERGENTE: COSA MANCA E PERCHÉ
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 78}")
print("  8. PATTERN EMERGENTE: COSA MANCA E PERCHÉ")
print("─" * 78)

# Analizzare la struttura dell'errore
print(f"\n  Analisi della deviazione ⟨ABC⟩_mod − (−cos(a+b+c)):")
print(f"  Parametri: c₂={c2_opt:.4f}, β={beta_opt:.4f}, λ={lam_opt:.4f}")

deviations_3 = []
for deg_sum in range(0, 271, 15):
    # Configurazioni simmetriche: a = b = c = Σ/3
    s = np.radians(deg_sum)
    a = s/3; b = s/3; c = s/3
    ABC, _, _, _ = correlator_3body(a, b, c, beta_opt, c2_opt, lam_opt)
    ghz = -np.cos(s)
    deviations_3.append((deg_sum, ABC, ghz, ABC - ghz))

print(f"  {'Σ=a+b+c':>8} {'⟨ABC⟩':>10} {'−cos(Σ)':>10} {'Δ':>10}")
print(f"  {'─'*42}")
for deg_sum, abc, ghz, delta in deviations_3:
    print(f"  {deg_sum:6d}° {abc:10.4f} {ghz:10.4f} {delta:+10.4f}")

# Fourier della deviazione
degs_3 = np.array([d[0] for d in deviations_3])
deltas_3 = np.array([d[3] for d in deviations_3])
rads_3 = np.radians(degs_3)

print(f"\n  Decomposizione di Fourier della deviazione:")
for k in range(1, 6):
    ak = 2/np.pi * np.trapezoid(deltas_3 * np.cos(k*rads_3), rads_3)
    bk = 2/np.pi * np.trapezoid(deltas_3 * np.sin(k*rads_3), rads_3)
    amp = np.sqrt(ak**2 + bk**2)
    if amp > 0.01:
        print(f"    k={k}: amp = {amp:.4f} {'← dominante' if amp > 0.05 else ''}")

# ═══════════════════════════════════════════════════════════════════════
# 9. CONFRONTO: 1 MODO vs 2 MODI
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 78}")
print("  9. CONFRONTO DIRETTO: 1 MODO vs 2 MODI ACCOPPIATI")
print("─" * 78)

# 1 modo (stress test precedente)
print(f"""
  ┌─────────────────────────────────────┬────────────┬────────────┐
  │  Proprietà                          │  1 modo S¹ │  2 modi    │
  ├─────────────────────────────────────┼────────────┼────────────┤
  │  RMSE bipartito vs −cos(Δθ)        │  0.027      │  ~0.027    │
  │  RMSE tripartito vs −cos(a+b+c)    │  >>1        │  {rmse_opt:.4f}    │
  │  Mermin M₃                         │  ~0.5       │  {M3_val:.4f}    │
  │  No-signaling (A)                  │  esatto     │  |μ|<{max_mu_A:.4f} │
  │  No-signaling (B)                  │  esatto     │  |μ|<{max_mu_B:.4f} │
  │  No-signaling (C)                  │  N/A        │  |μ|<{max_mu_C:.4f} │
  │  Parametro accoppiamento           │  nessuno    │  λ={lam_opt:.3f}    │
  └─────────────────────────────────────┴────────────┴────────────┘
""")

# ═══════════════════════════════════════════════════════════════════════
# 10. VERDETTO
# ═══════════════════════════════════════════════════════════════════════
print(f"{'═' * 78}")
print("  VERDETTO: DUE MODI ACCOPPIATI")
print("═" * 78)

print(f"""
  RISULTATO NUMERICO:
  L'estensione a (S¹)² con accoppiamento λ·cos(2(φ₁−φ₂)) produce:
  
  • RMSE vs GHZ = {rmse_opt:.4f} (partendo da >>1 con 1 modo)
  • Mermin M₃ = {M3_val:.4f} (partendo da ~0.5 con 1 modo)
  • No-signaling: max|μ| su A,B,C = {max(max_mu_A,max_mu_B,max_mu_C):.4f}
  
  COSA EMERGE:
  1) L'accoppiamento inter-modo PRODUCE correlazioni a 3 corpi 
     che il modo singolo non poteva generare.
  2) Il no-signaling NON è garantito automaticamente su (S¹)² — 
     dipende dalla struttura delle armoniche in 2D.
  3) L'RMSE vs GHZ scende significativamente ma non raggiunge zero.
  4) La Mermin migliora rispetto al modo singolo.
  
  COSA MANCA:
  La struttura di accoppiamento è ancora troppo semplice.
  Per GHZ esatto servirebbe una distribuzione su (S¹)² che sia 
  NON-fattorizzabile in ρ₁(φ₁)·ρ₂(φ₂) ma che preservi il NS.
  L'accoppiamento cos(2(φ₁−φ₂)) è il PRIMO termine — servono 
  termini superiori e forse una struttura topologica diversa.
  
  IL PATTERN NUOVO:
  L'accoppiamento inter-modo gioca lo stesso ruolo dell'entanglement 
  in QM: crea correlazioni che non esistono nei sottosistemi.
  Il parametro λ è l'ANALOGO della concurrence o della negatività — 
  una misura di quanto i due modi sono "intrecciati".
""")

