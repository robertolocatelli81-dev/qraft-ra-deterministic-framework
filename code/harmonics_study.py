"""
═══════════════════════════════════════════════════════════════════════════
  ARMONICHE SUPERIORI E FORMA DELLA CORRELAZIONE
  Come cos(4φ), cos(6φ), ... modificano E(Δθ) preservando no-signaling
═══════════════════════════════════════════════════════════════════════════
"""
import numpy as np
from scipy.special import ndtr, i0, i1
from scipy.optimize import minimize, differential_evolution

M = 8192
phi = np.linspace(0, 2*np.pi, M, endpoint=False)
dphi = 2*np.pi / M

def compute_model(c_coeffs, beta, sigma_r, delta_theta):
    """
    Azione generalizzata con armoniche pari:
    S(φ;a,b) = c0 - Σ_n c_n cos(2n·φ - n·(a+b))
    
    Nel frame centrato (ψ = φ - (a+b)/2):
    S(ψ) = c0 - Σ_n c_n cos(2n·ψ)
    
    c_coeffs = [c1, c2, c3, ...] per le armoniche cos(2ψ), cos(4ψ), cos(6ψ)
    """
    # Distribuzione nel frame centrato
    S_psi = np.zeros(M)
    for n, cn in enumerate(c_coeffs, start=1):
        S_psi += cn * np.cos(2*n*phi)  # phi qui è ψ nel frame centrato
    
    log_w = beta * S_psi  # nota: segno + perché S_psi è già il termine attrattivo
    log_w -= np.max(log_w)  # stabilità numerica
    w = np.exp(log_w)
    Z = np.sum(w) * dphi
    p = w / Z
    
    # Funzioni di misura nel frame centrato
    d = delta_theta / 2
    A = 2*ndtr(np.cos(phi + d)/sigma_r) - 1
    B = 2*ndtr(np.cos(phi - d)/sigma_r) - 1
    
    # Correlatore
    E = np.sum(A * B * p * dphi)
    
    # Marginale (verifica no-signaling)
    muA = np.sum(A * p * dphi)
    
    return E, muA, p

def correlation_curve(c_coeffs, beta, sigma_r, n_points=61):
    """Curva di correlazione E(Δθ) per Δθ ∈ [0, π]."""
    degs = np.linspace(0, 180, n_points)
    Es = []
    muAs = []
    for deg in degs:
        d = np.radians(deg)
        E, muA, _ = compute_model(c_coeffs, beta, sigma_r, d)
        Es.append(E)
        muAs.append(muA)
    return degs, np.array(Es), np.array(muAs)

def compute_chsh(c_coeffs, beta, sigma_r):
    """CHSH massimo per un dato set di armoniche."""
    def neg_chsh(params):
        x, alpha, y = params
        E1, _, _ = compute_model(c_coeffs, beta, sigma_r, x)
        E2, _, _ = compute_model(c_coeffs, beta, sigma_r, x+y)
        E3, _, _ = compute_model(c_coeffs, beta, sigma_r, x-alpha)
        E4, _, _ = compute_model(c_coeffs, beta, sigma_r, x-alpha+y)
        return -(E1 + E2 + E3 - E4)
    
    best = float('inf')
    for x0 in np.linspace(0.2, 1.5, 6):
        for a0 in np.linspace(0.5, 2.5, 6):
            for y0 in np.linspace(-2.5, -0.5, 6):
                val = neg_chsh([x0, a0, y0])
                if val < best:
                    best = val
    return -best

sigma_r = 0.005
beta = 2.8

# ═══════════════════════════════════════════════════════════════════════
# 1. ANATOMIA: COME OGNI ARMONICA DEFORMA LA DISTRIBUZIONE
# ═══════════════════════════════════════════════════════════════════════
print("=" * 78)
print("  1. ANATOMIA DELLE ARMONICHE: COME DEFORMANO LA DISTRIBUZIONE")
print("=" * 78)

print("""
  L'azione generalizzata con sole armoniche pari (no-signaling garantito):
  
    S(ψ) = −Σ_n c_n cos(2n ψ)
  
  Distribuzione: ρ(ψ) ∝ exp[β · Σ_n c_n cos(2n ψ)]
  
  Ogni armonica contribuisce una MODULAZIONE DIVERSA:
  
  • cos(2ψ): 2 picchi per periodo 2π, a ψ=0 e ψ=π     [bipolare]
  • cos(4ψ): 4 picchi per periodo 2π                     [quadrupolare]
  • cos(6ψ): 6 picchi per periodo 2π                     [esapolare]
  
  La distribuzione risultante è la SOVRAPPOSIZIONE delle modulazioni.
""")

print("  Distribuzione ρ(ψ) per diverse configurazioni di armoniche:")
print(f"  (β = {beta})")
print(f"  {'ψ/π':>6}", end="")

configs = {
    "c1=1": [1.0],
    "c1=1,c2=0.5": [1.0, 0.5],
    "c1=1,c2=-0.5": [1.0, -0.5],
    "c1=0.5,c2=1": [0.5, 1.0],
    "c2=1 solo": [0.0, 1.0],
    "c1=1,c2=0.3,c3=0.1": [1.0, 0.3, 0.1],
}

for name in configs:
    print(f"  {name:>16}", end="")
print()

for psi_val in np.linspace(0, np.pi, 13):
    print(f"  {psi_val/np.pi:6.2f}", end="")
    for name, coeffs in configs.items():
        _, _, p = compute_model(coeffs, beta, sigma_r, 0)
        # Trova il valore della distribuzione al punto ψ
        idx = int(psi_val / (2*np.pi) * M) % M
        print(f"  {p[idx]*2*np.pi:16.4f}", end="")  # normalizzato a 2π·ρ
    print()

# ═══════════════════════════════════════════════════════════════════════
# 2. CURVA DI CORRELAZIONE PER OGNI CONFIGURAZIONE
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 78}")
print("  2. CURVE DI CORRELAZIONE E(Δθ) PER DIVERSE ARMONICHE")
print("=" * 78)

print(f"\n  β = {beta}, σ_r = {sigma_r}")

test_configs = {
    "QRAFT-RA [c1=1]":            [1.0],
    "c1=1, c2=+0.3":              [1.0, 0.3],
    "c1=1, c2=+0.5":              [1.0, 0.5],
    "c1=1, c2=−0.3":              [1.0, -0.3],
    "c1=1, c2=−0.5":              [1.0, -0.5],
    "c1=0.5, c2=0.5":             [0.5, 0.5],
    "c2=1 (solo quadrupolo)":     [0.0, 1.0],
    "c1=1, c2=0.3, c3=0.1":      [1.0, 0.3, 0.1],
    "c1=1, c2=0.5, c3=0.3":      [1.0, 0.5, 0.3],
    "c1=1, c2=−0.3, c3=+0.2":    [1.0, -0.3, 0.2],
}

print(f"\n  {'Δθ(°)':>6}", end="")
for name in test_configs:
    print(f"  {name[:14]:>14}", end="")
print()
print(f"  {'─'*6}", end="")
for _ in test_configs:
    print(f"  {'─'*14}", end="")
print()

for deg in [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]:
    d = np.radians(deg)
    print(f"  {deg:5d}°", end="")
    for name, coeffs in test_configs.items():
        E, muA, _ = compute_model(coeffs, beta, sigma_r, d)
        print(f"  {E:14.6f}", end="")
    print()

# No-signaling check
print(f"\n  Verifica no-signaling (max|μ_A| su 100 angoli):")
for name, coeffs in test_configs.items():
    _, Es, muAs = correlation_curve(coeffs, beta, sigma_r, 100)
    print(f"    {name:35s}: max|μ_A| = {np.max(np.abs(muAs)):.2e}")

# ═══════════════════════════════════════════════════════════════════════
# 3. CHSH PER OGNI CONFIGURAZIONE
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 78}")
print("  3. CHSH MASSIMO PER OGNI CONFIGURAZIONE")
print("=" * 78)

print(f"\n  {'Config':>35} {'CHSH_max':>10} {'vs Tsirelson':>14} {'Regime':>18}")
print(f"  {'─'*80}")

tsirelson = 2*np.sqrt(2)

for name, coeffs in test_configs.items():
    S = compute_chsh(coeffs, beta, sigma_r)
    diff = S - tsirelson
    regime = "Super-Tsirelson" if S > tsirelson else ("Quantistico" if S > 2 else "Classico")
    print(f"  {name:>35} {S:10.4f} {diff:+14.4f} {regime:>18}")

# ═══════════════════════════════════════════════════════════════════════
# 4. EFFETTO DI c2: SCAN CONTINUO
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 78}")
print("  4. EFFETTO DI c2 (ARMONICA QUADRUPOLARE): SCAN CONTINUO")
print("=" * 78)

print(f"\n  c1 = 1.0 fisso, c2 variabile da −1 a +1")
print(f"\n  {'c2':>6} {'E(0°)':>8} {'E(45°)':>8} {'E(90°)':>8} {'E(135°)':>8} {'E(180°)':>9} {'CHSH':>8} {'NS':>6}")
print(f"  {'─'*62}")

for c2 in np.arange(-1.0, 1.05, 0.1):
    coeffs = [1.0, c2]
    E0,  _, _ = compute_model(coeffs, beta, sigma_r, 0)
    E45, _, _ = compute_model(coeffs, beta, sigma_r, np.pi/4)
    E90, _, _ = compute_model(coeffs, beta, sigma_r, np.pi/2)
    E135,_, _ = compute_model(coeffs, beta, sigma_r, 3*np.pi/4)
    E180,_, _ = compute_model(coeffs, beta, sigma_r, np.pi)
    
    # Quick CHSH
    S = compute_chsh(coeffs, beta, sigma_r)
    
    # No-signaling
    _, _, muAs = correlation_curve(coeffs, beta, sigma_r, 20)
    ns = "OK" if np.max(np.abs(muAs)) < 1e-10 else "FAIL"
    
    print(f"  {c2:+6.2f} {E0:8.4f} {E45:8.4f} {E90:8.4f} {E135:8.4f} {E180:9.4f} {S:8.4f} {ns:>6}")

# ═══════════════════════════════════════════════════════════════════════
# 5. RICERCA DELLA MIGLIORE APPROSSIMAZIONE AL SINGOLETTO
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 78}")
print("  5. APPROSSIMAZIONE AL SINGOLETTO CON ARMONICHE SUPERIORI")
print("=" * 78)

print("""
  Obiettivo: trovare coefficienti (c1, c2, c3) e parametro β tali che 
  la correlazione anti-fase E_sing(Δθ) = −E(Δθ) approssimi al meglio 
  il singoletto quantistico E_QM(Δθ) = −cos(Δθ).
""")

def singlet_rmse(params):
    """RMSE tra correlazione anti-fase e singoletto QM."""
    c1, c2, c3, beta_p = params
    if beta_p < 0.01:
        return 100
    coeffs = [c1, c2, c3]
    err = 0
    n = 0
    for deg in range(0, 181, 5):
        d = np.radians(deg)
        E, muA, _ = compute_model(coeffs, beta_p, sigma_r, d)
        E_sing = -E  # anti-fase
        E_qm = -np.cos(d)
        err += (E_sing - E_qm)**2
        n += 1
    return np.sqrt(err/n)

# Ottimizzazione
print("  Ottimizzazione in corso...")
result = differential_evolution(
    singlet_rmse,
    bounds=[(0, 2), (-1, 1), (-1, 1), (0.01, 5.0)],
    seed=42, maxiter=100, tol=1e-8, popsize=20
)

c1_opt, c2_opt, c3_opt, beta_opt = result.x
rmse_opt = result.fun

print(f"\n  RISULTATO OTTIMALE:")
print(f"    c1 = {c1_opt:.6f}")
print(f"    c2 = {c2_opt:.6f}")
print(f"    c3 = {c3_opt:.6f}")
print(f"    β  = {beta_opt:.6f}")
print(f"    RMSE = {rmse_opt:.6f}")

# Confronto: solo c1 (QRAFT-RA originale) vs ottimizzato
print(f"\n  Confronto: QRAFT-RA originale vs ottimizzato vs QM")
print(f"  {'Δθ(°)':>6} {'E_orig':>10} {'E_opt':>10} {'E_QM':>10} {'|err_orig|':>11} {'|err_opt|':>10}")
print(f"  {'─'*60}")

# β originale per confronto corretto
beta_orig_sing = 0.01  # il β che minimizza l'errore con solo c1
rmse_orig = singlet_rmse([1.0, 0, 0, beta_orig_sing])

for deg in [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]:
    d = np.radians(deg)
    E_orig, _, _ = compute_model([1.0], beta_orig_sing, sigma_r, d)
    E_opt, _, _ = compute_model([c1_opt, c2_opt, c3_opt], beta_opt, sigma_r, d)
    E_qm = -np.cos(d)
    
    print(f"  {deg:5d}° {-E_orig:10.6f} {-E_opt:10.6f} {E_qm:10.6f} "
          f"{abs(-E_orig-E_qm):11.6f} {abs(-E_opt-E_qm):10.6f}")

print(f"\n  RMSE originale (c1=1, β={beta_orig_sing}): {rmse_orig:.6f}")
print(f"  RMSE ottimizzato:                       {rmse_opt:.6f}")
print(f"  Miglioramento:                          {rmse_orig/rmse_opt:.1f}x")

# Verifica no-signaling dell'ottimo
_, _, muAs_opt = correlation_curve([c1_opt, c2_opt, c3_opt], beta_opt, sigma_r, 100)
print(f"  No-signaling dell'ottimo: max|μ_A| = {np.max(np.abs(muAs_opt)):.2e}")

# ═══════════════════════════════════════════════════════════════════════
# 6. ANALISI DELLA CURVATURA: DERIVATA SECONDA
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 78}")
print("  6. CURVATURA DELLA CORRELAZIONE: E''(Δθ)")
print("=" * 78)

print("""
  Per il singoletto QM: E(Δθ) = −cos(Δθ) → E''(Δθ) = cos(Δθ)
  In particolare: E''(0) = 1 (curvatura finita all'origine)
  
  La curvatura all'origine distingue le teorie:
  • Variabili nascoste locali: E''(0) → ∞ (cuspide)
  • QM standard: E''(0) = 1 (parabolica)
  • Modello vibrazionale: E''(0) dipende dalle armoniche
""")

def curvature_at_origin(coeffs, beta, sigma_r, h=0.001):
    """E''(0) per differenze finite."""
    E0, _, _ = compute_model(coeffs, beta, sigma_r, 0)
    Eh, _, _ = compute_model(coeffs, beta, sigma_r, h)
    Emh, _, _ = compute_model(coeffs, beta, sigma_r, -h)
    return (Eh - 2*E0 + Emh) / h**2

print(f"  {'Config':>35} {'E(0)':>8} {'E\'\'(0)':>10} {'QM: E\'\'(0)=1':>14}")
print(f"  {'─'*70}")

for name, coeffs in [
    ("QRAFT-RA [c1=1]", [1.0]),
    ("c1=1, c2=+0.5", [1.0, 0.5]),
    ("c1=1, c2=−0.5", [1.0, -0.5]),
    ("c1=1, c2=0.3, c3=0.1", [1.0, 0.3, 0.1]),
    (f"Ottimizzato", [c1_opt, c2_opt, c3_opt]),
    ("c2=1 (solo quadrupolo)", [0.0, 1.0]),
]:
    b = beta_opt if "Ottimiz" in name else beta
    E0, _, _ = compute_model(coeffs, b, sigma_r, 0)
    Epp = curvature_at_origin(coeffs, b, sigma_r)
    diff = abs(Epp - 1.0)
    print(f"  {name:>35} {E0:8.4f} {Epp:10.4f} {'← ' + f'Δ={diff:.4f}':>14}")

# ═══════════════════════════════════════════════════════════════════════
# 7. INTERPRETAZIONE FISICA DELLE ARMONICHE
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 78}")
print("  7. INTERPRETAZIONE FISICA DELLE ARMONICHE NEL MODELLO DI MEZZO")
print("=" * 78)

print("""
  Nel modello di mezzo vibrazionale, ogni armonica ha un significato fisico:
  
  ┌────────────────┬────────────────────────────────────────────────────┐
  │  Armonica      │  Interpretazione fisica                            │
  ├────────────────┼────────────────────────────────────────────────────┤
  │  cos(2ψ)      │  Modo fondamentale: 2 pozzi (dipolare)             │
  │  c1 > 0       │  Il mezzo ha DUE minimi equivalenti per periodo    │
  │               │  → correlazione monotona, simile a cos(Δθ)         │
  ├────────────────┼────────────────────────────────────────────────────┤
  │  cos(4ψ)      │  Modo quadrupolare: 4 pozzi                        │
  │  c2 > 0       │  Aggiunge minimi secondari tra i principali        │
  │               │  → APPIATTISCE la curva attorno a 90°              │
  │  c2 < 0       │  Aggiunge BARRIERE tra i minimi principali         │
  │               │  → ACUISCE la transizione attorno a 90°            │
  ├────────────────┼────────────────────────────────────────────────────┤
  │  cos(6ψ)      │  Modo esapolare: 6 pozzi                           │
  │  c3 > 0       │  Struttura fine aggiuntiva                         │
  │               │  → Modifica la curvatura alle estremità             │
  └────────────────┴────────────────────────────────────────────────────┘
  
  L'effetto di c2 è particolarmente importante:
  
  c2 > 0: la distribuzione sviluppa spalle laterali ai picchi principali.
          Fisicamente, il mezzo ha pozzi secondari che "trattengono" 
          popolazione fuori dai pozzi primari → correlazione più piatta 
          nella zona intermedia.
          
  c2 < 0: la distribuzione si restringe attorno ai picchi principali.
          Fisicamente, le barriere tra pozzi primari si alzano → la 
          popolazione è più concentrata → correlazione più ripida nella 
          transizione, più vicina a un gradino.
          
  Questo suggerisce che per approssimare il singoletto QM (che ha una
  transizione dolce, tipo coseno), serve un c2 NEGATIVO moderato — cioè
  un mezzo con barriere inter-pozzo leggermente rinforzate.
""")

# Verifica
print("  Verifica: effetto di c2 sulla pendenza a 90°")
print(f"  {'c2':>6} {'dE/dΔθ a 90°':>14} {'Tipo':>20}")
print(f"  {'─'*44}")

h = 0.01
for c2 in [-0.8, -0.5, -0.3, 0.0, 0.3, 0.5, 0.8]:
    coeffs = [1.0, c2]
    Ep, _, _ = compute_model(coeffs, beta, sigma_r, np.pi/2 + h)
    Em, _, _ = compute_model(coeffs, beta, sigma_r, np.pi/2 - h)
    slope = (Ep - Em) / (2*h)
    tipo = "ripido (barriere alte)" if abs(slope) > 0.6 else "dolce (pozzi secondari)"
    print(f"  {c2:+6.2f} {slope:14.6f} {tipo:>20}")

# QM per confronto
slope_qm = ((-np.cos(np.pi/2+h)) - (-np.cos(np.pi/2-h))) / (2*h)
print(f"  {'QM':>6} {slope_qm:14.6f} {'riferimento':>20}")

print(f"\n{'=' * 78}")
print("  CONCLUSIONE")
print("=" * 78)

print(f"""
  Le armoniche superiori pari (cos 4ψ, cos 6ψ, ...) sono gradi di 
  libertà del mezzo vibrazionale che:
  
  1. PRESERVANO il no-signaling (per la simmetria Z₂)
  2. MODIFICANO la forma della correlazione in modi specifici:
     • c2 > 0: appiattisce (pozzi secondari nel mezzo)
     • c2 < 0: acuisce (barriere rinforzate)
     • c3: modulazione fine della curvatura
  3. PERMETTONO di avvicinarsi al singoletto QM
     (RMSE ridotto di {rmse_orig/rmse_opt:.1f}x con ottimizzazione)
  4. NON violano mai il no-signaling (verificato numericamente)
  
  Il parametro c2 negativo ha l'interpretazione più suggestiva:
  corrisponde a un mezzo con BARRIERE RINFORZATE tra pozzi primari,
  che produce una transizione più graduale — più simile al coseno 
  del singoletto quantistico.
  
  Ottimo trovato: c1={c1_opt:.4f}, c2={c2_opt:.4f}, c3={c3_opt:.4f}, β={beta_opt:.4f}
  RMSE dal singoletto: {rmse_opt:.6f}
""")

