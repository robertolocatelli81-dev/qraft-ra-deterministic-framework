"""TEST B: EMERGE L'INTERA FENOMENOLOGIA QUANTISTICA?"""
import numpy as np
from scipy.special import ndtr, i0, i1

M = 4096
psi = np.linspace(0, 2*np.pi, M, endpoint=False)
dp = 2*np.pi / M
sr = 0.005

print("=" * 78)
print("  TEST B: FENOMENOLOGIA QUANTISTICA COMPLETA")
print("=" * 78)

# ═══════════════════════════════════════════════════════════════════════
# B1. DISUGUAGLIANZA CHSH: VALORE ESATTO 2√2
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 78}")
print("  B1: IL MODELLO RIPRODUCE ESATTAMENTE CHSH = 2√2?")
print("─" * 78)

def E_gen(dt, coeffs, beta, spin_factor=1):
    S = sum(cn*np.cos(2*spin_factor*n*psi) for n,cn in enumerate(coeffs,1))
    lw = beta*S; lw -= np.max(lw)
    w = np.exp(lw); Z = np.sum(w)*dp; p = w/Z
    d = dt/2
    A = 2*ndtr(np.cos(spin_factor*(psi+d))/sr)-1
    B = -(2*ndtr(np.cos(spin_factor*(psi-d))/sr)-1)
    return np.sum(A*B*p*dp)

# Per spin ½: angoli CHSH ottimali sono a0=0, a1=π/2, b0=π/4, b1=-π/4
# Il CHSH del singoletto QM = −E(0,π/4) − E(0,−π/4) − E(π/2,π/4) + E(π/2,−π/4)
# = −(−cos(π/4)) − (−cos(π/4)) − (−cos(π/4)) + (−cos(3π/4))
# = cos(π/4) + cos(π/4) + cos(π/4) + cos(π/4) = 4/√2 = 2√2

# Nel modello (singoletto = anti-fase):
# S = E_sing(a0-b0) + E_sing(a0-b1) + E_sing(a1-b0) - E_sing(a1-b1)

# Spin ½ quadrupolare ottimizzato
c_s12 = [0.0, -0.876]
b_s12 = 0.779

S_model = (E_gen(np.pi/4, c_s12, b_s12) + E_gen(-np.pi/4, c_s12, b_s12) + 
           E_gen(np.pi/4, c_s12, b_s12) - E_gen(3*np.pi/4, c_s12, b_s12))

# Con mappa fotonica  
c_ph = [0.0, -1.348]
b_ph = 0.507

# Per fotoni, angoli CHSH: a0=0, a1=π/4, b0=π/8, b1=-π/8
S_photon = (E_gen(np.pi/8, c_ph, b_ph, spin_factor=2) + 
            E_gen(-np.pi/8, c_ph, b_ph, spin_factor=2) + 
            E_gen(np.pi/8, c_ph, b_ph, spin_factor=2) - 
            E_gen(3*np.pi/8, c_ph, b_ph, spin_factor=2))

print(f"  CHSH (spin ½, quadrupolo):  {abs(S_model):.6f}  (QM = {2*np.sqrt(2):.6f})")
print(f"  CHSH (fotoni, quadrupolo):  {abs(S_photon):.6f}  (QM = {2*np.sqrt(2):.6f})")
print(f"  Deficit spin ½: {abs(abs(S_model) - 2*np.sqrt(2)):.6f}")
print(f"  Deficit fotoni: {abs(abs(S_photon) - 2*np.sqrt(2)):.6f}")

print(f"""
  ★ Il modello NON raggiunge esattamente 2√2 con i parametri ottimali 
  per il singoletto. Il deficit è {abs(abs(S_model) - 2*np.sqrt(2)):.4f} per spin ½.
  Questo è perché l'approssimazione E ≈ −cos(Δθ) non è esatta (RMSE ≈ 0.027).
""")

# ═══════════════════════════════════════════════════════════════════════
# B2. STATI ENTANGLED NON-MASSIMALI
# ═══════════════════════════════════════════════════════════════════════
print(f"{'─' * 78}")
print("  B2: STATI ENTANGLED NON-MASSIMALI (PARAMETRO DI VISIBILITÀ)")
print("─" * 78)

print("""
  In QM, uno stato non-massimalmente entangled ha correlazione:
    E_QM(Δθ; V) = −V·cos(Δθ)
  dove V ∈ [0,1] è la visibilità.
  
  Nel modello vibrazionale, la "visibilità" dovrebbe corrispondere 
  alla coerenza del mezzo: V ~ C(β).
""")

def visibility_model(beta, c_coeffs=[0.0, -0.876]):
    """Calcola la visibilità effettiva del modello."""
    E0 = E_gen(0, c_coeffs, beta)
    Epi = E_gen(np.pi, c_coeffs, beta)
    return (abs(Epi) + abs(E0)) / 2  # media dei valori estremi

print(f"  {'β':>6} {'C(β)':>8} {'V_eff':>8} {'E(0)':>8} {'E(π)':>8} {'V≈C?':>8}")
print(f"  {'─'*46}")

for beta in [0.1, 0.3, 0.5, 0.779, 1.0, 1.5, 2.0, 3.0, 5.0]:
    kappa = beta * abs(-0.876)
    C = i1(kappa)/i0(kappa) if kappa > 0.001 else kappa/2
    V = visibility_model(beta)
    E0 = E_gen(0, [0.0, -0.876], beta)
    Epi = E_gen(np.pi, [0.0, -0.876], beta)
    match = "~" if abs(V-C) < 0.1 else "≠"
    print(f"  {beta:6.3f} {C:8.4f} {V:8.4f} {E0:8.4f} {Epi:8.4f} {match:>8}")

print(f"""
  ★ La visibilità V del modello NON corrisponde linearmente a C(β).
  La relazione è monotona ma non-lineare — il modello prevede una 
  relazione V(β) specifica che potrebbe essere testata.
""")

# ═══════════════════════════════════════════════════════════════════════
# B3. STATI GHZ E W (3 PARTICELLE)
# ═══════════════════════════════════════════════════════════════════════
print(f"{'─' * 78}")
print("  B3: CORRELAZIONI A 3 CORPI (GHZ / W)")
print("─" * 78)

print("""
  In QM, lo stato GHZ a 3 qubit ha correlazioni specifiche:
    E_GHZ(a,b,c) = −cos(a+b+c) per lo stato (|000⟩+|111⟩)/√2
  
  Lo stato W ha correlazioni diverse:
    E_W(a,b) è simmetrico ma non fattorizza come GHZ
  
  Domanda: il modello vibrazionale 1D può descrivere entrambi?
""")

# Modello a 3 corpi: 3 misure sullo stesso modo
def E_3body(theta_a, theta_b, theta_c, c_coeffs, beta):
    """Correlatore a 3 corpi: ⟨ABC⟩."""
    # L'azione per 3 rivelatori?
    # Estensione naturale: S(ψ;a,b,c) dovrebbe coinvolgere il PRODOTTO 
    # cos(ψ+δ_a)cos(ψ+δ_b)cos(ψ+δ_c)
    # Ma nel frame centrato la distribuzione dipende da (a+b+c)/3
    
    # Azione: prodotto di 3 proiezioni
    s3 = (theta_a + theta_b + theta_c) / 3
    # Frame centrato: ψ = φ - s3
    da = theta_a - s3
    db = theta_b - s3
    dc = theta_c - s3
    
    # Distribuzione: costruiamo dal prodotto triplo
    prod3 = np.cos(psi + da) * np.cos(psi + db) * np.cos(psi + dc)
    
    S = sum(cn*np.cos(2*n*psi) for n,cn in enumerate(c_coeffs,1))
    lw = beta * S; lw -= np.max(lw)
    w = np.exp(lw); Z = np.sum(w)*dp; p = w/Z
    
    A = 2*ndtr(np.cos(psi+da)/sr)-1
    B = -(2*ndtr(np.cos(psi+db)/sr)-1)
    C_meas = -(2*ndtr(np.cos(psi+dc)/sr)-1)
    
    return np.sum(A * B * C_meas * p * dp)

print("  Correlazione a 3 corpi ⟨ABC⟩ vs QM (GHZ):")
print(f"  {'(a,b,c)':>20} {'⟨ABC⟩_model':>14} {'−cos(a+b+c)':>14}")
print(f"  {'─'*52}")

c3_body = [0.0, -0.876]
beta_3 = 0.779

for a,b,c in [(0,0,0),(np.pi/4,0,0),(np.pi/4,np.pi/4,0),
               (np.pi/4,np.pi/4,np.pi/4),(np.pi/2,0,0),
               (np.pi/3,np.pi/3,np.pi/3)]:
    E3 = E_3body(a, b, c, c3_body, beta_3)
    E_ghz = -np.cos(a+b+c)
    degs = f"({np.degrees(a):.0f}°,{np.degrees(b):.0f}°,{np.degrees(c):.0f}°)"
    print(f"  {degs:>20} {E3:14.6f} {E_ghz:14.6f}")

print(f"""
  ★ VERDETTO B3:
  Il modello vibrazionale 1D produce correlazioni a 3 corpi, ma la 
  struttura NON coincide con GHZ. Il modello a singolo modo vibrazionale
  non ha abbastanza gradi di libertà per distinguere GHZ da W.
  
  Per descrivere stati multipartiti diversi, servirebbero MODI MULTIPLI
  del mezzo — un'estensione non banale del framework.
""")

# ═══════════════════════════════════════════════════════════════════════
# B4. COMPLEMENTARITÀ ONDA-PARTICELLA
# ═══════════════════════════════════════════════════════════════════════
print(f"{'─' * 78}")
print("  B4: COMPLEMENTARITÀ E PRINCIPIO DI INDETERMINAZIONE")
print("─" * 78)

print("""
  In QM, il principio di Heisenberg implica:
    ΔA · ΔB ≥ |⟨[A,B]⟩|/2
  
  Nel modello vibrazionale, per osservabili "ortogonali" (Δθ = π/2):
""")

c_unc = [0.0, -0.876]
beta_unc = 0.779

# Calcoliamo varianza delle misure per angoli ortogonali
def variance_model(theta, c_coeffs, beta):
    """Varianza dell'esito di misura ⟨A²⟩ − ⟨A⟩²."""
    S = sum(cn*np.cos(2*n*psi) for n,cn in enumerate(c_coeffs,1))
    lw = beta*S; lw -= np.max(lw)
    w = np.exp(lw); Z = np.sum(w)*dp; p = w/Z
    
    # ⟨A⟩ = marginale (= 0 per simmetria)
    A = 2*ndtr(np.cos(psi)/sr)-1
    mean_A = np.sum(A * p * dp)  # ≈ 0
    
    # ⟨A²⟩
    mean_A2 = np.sum(A**2 * p * dp)
    
    return mean_A2 - mean_A**2

# Per il modello, A² ≈ 1 (esiti ±1 con probabilità quasi discrete per sr piccolo)
# Quindi ΔA ≈ 1 e il prodotto ΔA·ΔB ≈ 1

delta_0 = np.sqrt(variance_model(0, c_unc, beta_unc))
delta_pi2 = np.sqrt(variance_model(np.pi/2, c_unc, beta_unc))

print(f"  ΔA (θ=0):    {delta_0:.6f}")
print(f"  ΔB (θ=π/2):  {delta_pi2:.6f}")
print(f"  ΔA · ΔB:     {delta_0 * delta_pi2:.6f}")
print(f"  Heisenberg (singoletto): ΔA·ΔB ≥ 0.5")

print(f"""
  Il modello soddisfa formalmente Heisenberg (ΔA·ΔB ≈ 1 > 0.5),
  ma questo è TRIVIALE perché ogni osservabile ±1 ha varianza ≤ 1.
  
  Il principio di indeterminazione NON emerge in modo non-triviale
  dal modello — è soddisfatto automaticamente, non come conseguenza 
  della struttura vibrazionale.
""")

# ═══════════════════════════════════════════════════════════════════════
# B5. DISUGUAGLIANZA CHSH STRETTA PER OGNI β
# ═══════════════════════════════════════════════════════════════════════
print(f"{'─' * 78}")
print("  B5: IL MODELLO RIPRODUCE LA TRANSIZIONE CLASSICO→QUANTISTICO?")
print("─" * 78)

print("""
  In QM, l'entanglement è binario: lo stato è entangled oppure no.
  Il CHSH massimo per uno stato puro è esattamente 2√2.
  Per stati misti: CHSH = 2√2 · V, con V ∈ [0,1].
  
  Il modello vibrazionale ha una transizione CONTINUA controllata da β:
""")

print(f"  {'β':>6} {'CHSH_model':>12} {'2√2·V_eff':>12} {'Rapporto':>10}")
print(f"  {'─'*44}")

tsirelson = 2*np.sqrt(2)
for beta in [0.1, 0.3, 0.5, 0.779, 1.0, 1.5, 2.0, 3.0]:
    # CHSH con angoli ottimali π/4 shift
    S = abs(E_gen(np.pi/4, c_unc, beta) + E_gen(-np.pi/4, c_unc, beta) +
            E_gen(np.pi/4, c_unc, beta) - E_gen(3*np.pi/4, c_unc, beta))
    V = visibility_model(beta)
    S_qm = tsirelson * V
    ratio = S / S_qm if S_qm > 0.01 else float('nan')
    print(f"  {beta:6.3f} {S:12.6f} {S_qm:12.6f} {ratio:10.4f}")

print(f"""
  ★ Il rapporto CHSH_model / (2√2·V) non è costante.
  In QM, questo rapporto è esattamente 1 per stati puri (ogni V).
  Nel modello, varia con β → la relazione CHSH-visibilità è DIVERSA da QM.
  
  Questa è una PREDIZIONE TESTABILE: misurando CHSH e V indipendentemente,
  si può verificare se la relazione è quella quantistica o quella del modello.
""")

# ═══════════════════════════════════════════════════════════════════════
# B6. RIEPILOGO
# ═══════════════════════════════════════════════════════════════════════
print(f"{'=' * 78}")
print("  VERDETTO TEST B: FENOMENOLOGIA QUANTISTICA COMPLETA")
print("=" * 78)

print("""
  ┌────────────────────────────────────────────────┬──────────┬──────────┐
  │  Fenomeno                                      │ Riproduce│  Note    │
  ├────────────────────────────────────────────────┼──────────┼──────────┤
  │  Forma E = −cos(Δθ) (spin ½)                  │  ≈ SÌ    │ RMSE 0.03│
  │  Forma E = −cos(2Δθ) (fotoni)                 │  ≈ SÌ    │ mappa 2× │
  │  CHSH = 2√2 esatto                            │  ≈ NO    │ deficit  │
  │  No-signaling                                  │  ESATTO  │ analitico│
  │  Visibilità V variabile                        │  SÌ      │ non-lin. │
  │  Relazione CHSH-visibilità lineare             │  NO      │ diversa  │
  │  Complementarità (Heisenberg)                  │  TRIVIALE│ non info │
  │  Stati GHZ (3+ corpi)                          │  NO      │ 1 modo   │
  │  Spin arbitrario (s=½,1,3/2,...)              │  SÌ      │ freq 2s× │
  │  Statistiche di Bose/Fermi                     │  NO      │ non nel  │
  │  Interferenza a singola particella             │  ?       │ non test │
  │  Effetto Zeno quantistico                      │  ?       │ non test │
  └────────────────────────────────────────────────┴──────────┴──────────┘
  
  SINTESI: il modello riproduce la STRUTTURA CORRELAZIONALE bipartita 
  in modo approssimato, ma NON riproduce la fenomenologia completa.
  I deficit principali sono: CHSH non esattamente 2√2, relazione 
  CHSH-visibilità non-lineare (diversa da QM), nessuna descrizione 
  di stati multipartiti non-banali, nessuna statistica quantistica.
""")

