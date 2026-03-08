"""
═══════════════════════════════════════════════════════════════════════════
  STRESS TEST II: FENOMENOLOGIA MULTIPARTITA (GHZ, W, INTERFERENZA)
  E STRESS TEST III: STATISTICHE QUANTISTICHE
═══════════════════════════════════════════════════════════════════════════
"""
import numpy as np
from scipy.special import ndtr, i0, i1

M = 4096
psi = np.linspace(0, 2*np.pi, M, endpoint=False)
dp = 2*np.pi / M; sr = 0.005
c2 = -0.876; beta0 = 0.779

print("=" * 78)
print("  STRESS TEST II: STATI MULTIPARTITI")
print("=" * 78)

# ═══════════════════════════════════════════════════════════════════════
# II.1 ESTENSIONE NATURALE A 3 CORPI: MODO CONDIVISO
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 78}")
print("  II.1 TRE PARTICELLE SU UN MODO CONDIVISO")
print("─" * 78)

print("""
  Approccio: 3 rivelatori (a, b, c) perturbano lo STESSO mezzo.
  L'azione a 3 corpi naturale è il PRODOTTO TRIPLO:
  
    S₃(φ; a, b, c) = cos(φ−a) · cos(φ−b) · cos(φ−c)
  
  Espandiamo con identità trigonometriche:
  cos(X)cos(Y)cos(Z) = [cos(X−Y−Z) + cos(X+Y−Z) + cos(X−Y+Z) + cos(X+Y+Z)] / 4
""")

def expand_triple_product(a, b, c):
    """Decomposizione di Fourier del prodotto triplo."""
    vals = np.cos(psi-a)*np.cos(psi-b)*np.cos(psi-c)
    # Coefficienti di Fourier
    coeffs = {}
    for k in range(8):
        ck_cos = 2*np.sum(vals*np.cos(k*psi))*dp/(2*np.pi)
        ck_sin = 2*np.sum(vals*np.sin(k*psi))*dp/(2*np.pi)
        amp = np.sqrt(ck_cos**2 + ck_sin**2)
        if amp > 1e-6:
            coeffs[k] = (ck_cos, ck_sin, amp)
    return coeffs

# Analisi per configurazione simmetrica
print("  Armoniche del prodotto triplo (a=0, b=2π/3, c=4π/3):")
coeffs = expand_triple_product(0, 2*np.pi/3, 4*np.pi/3)
for k, (cc, cs, amp) in sorted(coeffs.items()):
    print(f"    k={k}: amp = {amp:.6f}, "
          f"{'cos' if abs(cc) > abs(cs) else 'sin'}({k}ψ)")

print("""
  ★ Il prodotto triplo contiene armoniche di frequenza 1 E 3.
  Frequenza 1 è DISPARI → ROMPE il no-signaling!
  
  Questo è un problema strutturale: l'accoppiamento a 3 corpi 
  tramite prodotto triplo NON preserva automaticamente il NS.
""")

# Verifica NS per 3 corpi
def E3_model(a, b, c, beta):
    """Correlatore a 3 corpi su modo condiviso."""
    # Azione: prodotto triplo
    S = np.cos(psi-a)*np.cos(psi-b)*np.cos(psi-c)
    lw = beta*S; lw -= np.max(lw)
    w = np.exp(lw); Z = np.sum(w)*dp; p = w/Z
    
    A_ = 2*ndtr(np.cos(psi-a)/sr)-1
    B_ = -(2*ndtr(np.cos(psi-b)/sr)-1)
    C_ = -(2*ndtr(np.cos(psi-c)/sr)-1)
    
    ABC = np.sum(A_*B_*C_*p*dp)
    muA = np.sum(A_*p*dp)
    return ABC, muA

# NS check
max_mu3 = 0
for a in np.linspace(0, np.pi, 5):
    for b in np.linspace(0, np.pi, 5):
        for c in np.linspace(0, np.pi, 5):
            _, mu = E3_model(a, b, c, 1.0)
            max_mu3 = max(max_mu3, abs(mu))

print(f"  NS con prodotto triplo: max|μ_A| = {max_mu3:.4f}")
print(f"  → {'ROTTO!' if max_mu3 > 0.01 else 'OK'}")

# ═══════════════════════════════════════════════════════════════════════
# II.2 ALTERNATIVA: ACCOPPIAMENTI A COPPIE
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 78}")
print("  II.2 ALTERNATIVA: MODI MULTIPLI (uno per coppia)")
print("─" * 78)

print("""
  Se un singolo modo non basta, possiamo usare MODI MULTIPLI.
  
  Per 3 particelle (A, B, C):
  • Modo 1: condiviso da A-B (fase φ₁)
  • Modo 2: condiviso da A-C (fase φ₂)  
  • Modo 3: condiviso da B-C (fase φ₃)
  
  Ogni coppia ha il suo modo vibrazionale → bipartito per costruzione.
  Le correlazioni a 3 corpi emergono dalle INTERSEZIONI dei modi.
  
  GHZ: tutti e 3 i modi sono correlati (accoppiamento inter-modo)
  W:   i modi sono indipendenti (nessun accoppiamento inter-modo)
""")

# Modello multi-modo semplice: 2 modi indipendenti
def E3_multi_mode(a, b, c, beta):
    """3 corpi con 3 modi indipendenti (uno per coppia)."""
    # Modo AB
    S_ab = c2*np.cos(4*psi)
    lw_ab = beta*S_ab; lw_ab -= np.max(lw_ab)
    w_ab = np.exp(lw_ab); Z_ab = np.sum(w_ab)*dp; p_ab = w_ab/Z_ab
    
    d_ab = (b-a)/2
    A_ab = 2*ndtr(np.cos(psi+d_ab)/sr)-1
    B_ab = -(2*ndtr(np.cos(psi-d_ab)/sr)-1)
    E_ab = np.sum(A_ab*B_ab*p_ab*dp)
    
    # Modo AC (stesso tipo, diversa differenza)
    d_ac = (c-a)/2
    A_ac = 2*ndtr(np.cos(psi+d_ac)/sr)-1
    C_ac = -(2*ndtr(np.cos(psi-d_ac)/sr)-1)
    E_ac = np.sum(A_ac*C_ac*p_ab*dp)  # stessa distribuzione
    
    # Modo BC
    d_bc = (c-b)/2
    B_bc = 2*ndtr(np.cos(psi+d_bc)/sr)-1
    C_bc = -(2*ndtr(np.cos(psi-d_bc)/sr)-1)
    E_bc = np.sum(B_bc*C_bc*p_ab*dp)
    
    return E_ab, E_ac, E_bc

# GHZ: ⟨ABC⟩ = −cos(a+b+c) per il vero GHZ
# Nel modello multi-modo: ⟨ABC⟩ ≈ prodotto delle correlazioni? No.
# Per modi indipendenti: ⟨ABC⟩ non è definito semplicemente.

print("  Correlazioni a coppie per 3 particelle (modi indipendenti):")
print(f"  {'(a,b,c)':>25} {'E_AB':>8} {'E_AC':>8} {'E_BC':>8} "
      f"{'GHZ: −cos(Σ)':>12}")
print(f"  {'─'*68}")

for a,b,c in [(0,0,0), (np.pi/4,np.pi/4,np.pi/4), 
               (0,np.pi/4,np.pi/2), (np.pi/6,np.pi/3,np.pi/2)]:
    E_ab, E_ac, E_bc = E3_multi_mode(a, b, c, beta0)
    ghz = -np.cos(a+b+c)
    degs = f"({np.degrees(a):.0f}°,{np.degrees(b):.0f}°,{np.degrees(c):.0f}°)"
    print(f"  {degs:>25} {E_ab:8.4f} {E_ac:8.4f} {E_bc:8.4f} {ghz:12.4f}")

print(f"""
  ★ I modi indipendenti producono correlazioni a coppie consistenti,
  ma NON producono la correlazione a 3 corpi GHZ ⟨ABC⟩ = −cos(a+b+c).
  
  Per ottenere GHZ servirebbe un ACCOPPIAMENTO TRA I MODI.
  Questa è un'estensione non-banale del framework 1D.
""")

# ═══════════════════════════════════════════════════════════════════════
# II.3 TEST CRITICO: DISUGUAGLIANZA DI MERMIN
# ═══════════════════════════════════════════════════════════════════════
print(f"{'─' * 78}")
print("  II.3 DISUGUAGLIANZA DI MERMIN (3 PARTICELLE)")
print("─" * 78)

print("""
  Per 3 qubit, la disuguaglianza di Mermin è:
    M₃ = |⟨A₁B₂C₂⟩ + ⟨A₂B₁C₂⟩ + ⟨A₂B₂C₁⟩ − ⟨A₁B₁C₁⟩| ≤ 2
  
  QM (stato GHZ): M₃ = 4
  
  Nel modello a modo singolo con prodotto triplo:
""")

def mermin_single_mode(beta):
    """Disuguaglianza di Mermin per modo singolo."""
    # Impostazioni: x₁=0, x₂=π/2 per ogni parte
    settings = [(0, np.pi/2)]  # 2 scelte per particella
    
    def corr3(a, b, c, beta):
        S = np.cos(psi-a)*np.cos(psi-b)*np.cos(psi-c)
        S = S - np.min(S)  # shift
        lw = beta*S; lw -= np.max(lw)
        w = np.exp(lw); Z = np.sum(w)*dp; p = w/Z
        A_ = 2*ndtr(np.cos(psi-a)/sr)-1
        B_ = -(2*ndtr(np.cos(psi-b)/sr)-1)
        C_ = -(2*ndtr(np.cos(psi-c)/sr)-1)
        return np.sum(A_*B_*C_*p*dp)
    
    a1, a2 = 0, np.pi/2
    b1, b2 = 0, np.pi/2
    c1, c2 = 0, np.pi/2
    
    M3 = abs(corr3(a1,b2,c2,beta) + corr3(a2,b1,c2,beta) + 
             corr3(a2,b2,c1,beta) - corr3(a1,b1,c1,beta))
    return M3

for b in [0.5, 0.779, 1.0, 2.0, 5.0]:
    M3 = mermin_single_mode(b)
    print(f"  β={b:5.3f}: M₃ = {M3:.6f}  (classico ≤ 2, GHZ = 4)")

print(f"""
  ★ Il modello a modo singolo NON raggiunge M₃ = 4 (GHZ).
  Produce valori ben sotto il limite classico di 2.
  Questo conferma: un singolo modo 1D non ha abbastanza 
  struttura per gli stati genuinamente multipartiti.
""")

# ═══════════════════════════════════════════════════════════════════════
# TEST III: STATISTICHE QUANTISTICHE E INTERFERENZA
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 78}")
print("  STRESS TEST III: STATISTICHE QUANTISTICHE E INTERFERENZA")
print("═" * 78)

# ═══════════════════════════════════════════════════════════════════════
# III.1 INTERFERENZA A SINGOLA PARTICELLA
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 78}")
print("  III.1 INTERFERENZA A DOPPIA FENDITURA (SINGOLA PARTICELLA)")
print("─" * 78)

print("""
  In QM, una singola particella attraverso una doppia fenditura 
  produce una figura di interferenza:
  
    P(x) ∝ |ψ₁(x) + ψ₂(x)|² = |ψ₁|² + |ψ₂|² + 2Re(ψ₁*ψ₂)
  
  Il terzo termine è l'INTERFERENZA.
  
  Nel modello vibrazionale, la particella segue le onde del mezzo.
  Se il mezzo passa attraverso due fenditure, le onde interferiscono.
  La distribuzione della particella dovrebbe mostrare frange.
  
  MODELLO: due sorgenti puntuali nel mezzo a posizioni θ₁, θ₂ su S¹.
  La distribuzione è la sovrapposizione dei due modi:
""")

def interference_pattern(theta1, theta2, beta, n_points=100):
    """Pattern di interferenza di due sorgenti nel mezzo."""
    x = np.linspace(0, 2*np.pi, n_points)
    # Onda dalla sorgente 1: A₁ exp[iκ(x−θ₁)] → nel mezzo: cos(κ(x−θ₁))
    # Onda dalla sorgente 2: A₂ exp[iκ(x−θ₂)] → nel mezzo: cos(κ(x−θ₂))
    # Intensità: |A₁ cos(κ(x−θ₁)) + A₂ cos(κ(x−θ₂))|²
    
    kappa = 2  # frequenza fondamentale
    wave1 = np.cos(kappa*(x - theta1))
    wave2 = np.cos(kappa*(x - theta2))
    
    # Intensità classica (senza interferenza): |w1|² + |w2|²
    I_classical = wave1**2 + wave2**2
    
    # Intensità con interferenza: |w1 + w2|²
    I_quantum = (wave1 + wave2)**2
    
    # Nel modello di mezzo: la distribuzione è Gibbs con azione dalle due sorgenti
    S_mezzo = -(wave1 + wave2)**2  # energia = −intensità
    lw = beta * S_mezzo / np.max(np.abs(S_mezzo))
    lw -= np.max(lw)
    w = np.exp(lw)
    I_medium = w / np.sum(w) * len(x) / (2*np.pi)
    
    return x, I_classical, I_quantum, I_medium

x, I_cl, I_qm, I_med = interference_pattern(np.pi/3, 2*np.pi/3, 2.0)

# Visibilità delle frange
V_qm = (np.max(I_qm) - np.min(I_qm)) / (np.max(I_qm) + np.min(I_qm))
V_med = (np.max(I_med) - np.min(I_med)) / (np.max(I_med) + np.min(I_med))
V_cl = (np.max(I_cl) - np.min(I_cl)) / (np.max(I_cl) + np.min(I_cl))

print(f"  Visibilità delle frange:")
print(f"    Classico (no interferenza): V = {V_cl:.4f}")
print(f"    QM (interferenza piena):    V = {V_qm:.4f}")  
print(f"    Mezzo vibrazionale:         V = {V_med:.4f}")

# Il pattern del mezzo ha frange?
n_maxima_med = 0
for i in range(1, len(I_med)-1):
    if I_med[i] > I_med[i-1] and I_med[i] > I_med[i+1]:
        n_maxima_med += 1

n_maxima_qm = 0
for i in range(1, len(I_qm)-1):
    if I_qm[i] > I_qm[i-1] and I_qm[i] > I_qm[i+1]:
        n_maxima_qm += 1

print(f"    Numero di massimi (QM):     {n_maxima_qm}")
print(f"    Numero di massimi (mezzo):  {n_maxima_med}")

print(f"""
  ★ Il modello di mezzo PRODUCE interferenza (V > 0), ma la forma 
  è diversa dalla QM standard. Le frange esistono perché il mezzo 
  supporta la sovrapposizione di onde, ma la distribuzione di Gibbs 
  non è uguale a |ψ₁+ψ₂|².
  
  La differenza è che nel mezzo, l'intensità è filtrata da una 
  distribuzione termica (exp[−βE]) anziché essere quadratica (|ψ|²).
""")

# ═══════════════════════════════════════════════════════════════════════
# III.2 BOSONI E FERMIONI: SIMMETRIA DI SCAMBIO
# ═══════════════════════════════════════════════════════════════════════
print(f"{'─' * 78}")
print("  III.2 STATISTICHE DI BOSE-EINSTEIN E FERMI-DIRAC")
print("─" * 78)

print("""
  In QM, la funzione d'onda di 2 bosoni è SIMMETRICA:
    ψ(φ₁, φ₂) = ψ(φ₂, φ₁)       [bunching]
  
  Per 2 fermioni è ANTISIMMETRICA:
    ψ(φ₁, φ₂) = −ψ(φ₂, φ₁)      [anti-bunching, Pauli]
  
  Nel modello vibrazionale con UN modo su S¹:
  • La fase φ è UNICA (condivisa) → non c'è distinzione φ₁, φ₂
  • Due particelle sullo stesso modo hanno la STESSA fase
  • Non c'è meccanismo naturale per l'anti-simmetria (Fermi)
  
  Per ottenere statistiche quantistiche, servirebbero:
  • Modi MULTIPLI (φ₁, φ₂ indipendenti su copie di S¹)
  • Una regola di simmetrizzazione/antisimmetrizzazione
  • Questa regola è AGGIUNTIVA — non emerge dal framework 1D
""")

# Test: due particelle sullo stesso modo — bunching o anti-bunching?
def two_particle_correlation(delta_phi, beta):
    """Probabilità congiunta P(φ₁, φ₂) per due particelle sullo stesso modo."""
    # Se condividono lo stesso modo: P(φ₁,φ₂) = ρ(φ₁) δ(φ₁−φ₂)
    # → perfetta correlazione, nessun bunching/anti-bunching
    # Questo è TROPPO: non distingue bosoni da fermioni
    return 1.0  # triviale

print("  Due particelle sullo stesso modo:")
print("  → Correlazione triviale (stessa fase)")
print("  → Nessuna distinzione bosone/fermione")
print("  → Il framework 1D NON produce statistiche quantistiche")

# ═══════════════════════════════════════════════════════════════════════
# III.3 EFFETTO HONG-OU-MANDEL
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'─' * 78}")
print("  III.3 EFFETTO HONG-OU-MANDEL")
print("─" * 78)

print("""
  Nell'effetto HOM, due fotoni identici che arrivano su un 
  beamsplitter 50:50 escono SEMPRE dalla stessa porta (bunching).
  
  P(coincidenza) = 0  per fotoni identici (QM: interferenza distruttiva)
  P(coincidenza) = 0.5 per fotoni distinguibili (classico)
  
  Nel modello vibrazionale:
  • Due fotoni "sullo stesso modo" condividono la stessa fase
  • Un beamsplitter mescola due modi
  • Per il bunching serve che P(porte diverse) = 0
  
  Con un singolo modo 1D, il beamsplitter non è definibile:
  servirebbero almeno 2 modi (uno per porta di ingresso).
""")

# Tentativo con 2 modi
def hom_two_modes(delta_t, beta):
    """Effetto HOM con 2 modi vibrazionali."""
    # Modo 1: fotone A, modo 2: fotone B
    # Beamsplitter: φ_out1 = (φ₁+φ₂)/2, φ_out2 = (φ₁−φ₂)/2
    # Per fotoni identici: φ₁ ≈ φ₂ → φ_out2 ≈ 0
    
    # Distribuzione congiunta: ρ(φ₁,φ₂) = ρ(φ₁)ρ(φ₂) per modi indipendenti
    # Dopo BS: ρ(φ_out1, φ_out2)
    
    # Ritardo temporale δt → sfasamento δφ
    delta_phi = delta_t  # in unità angolari
    
    # Probabilità di coincidenza: P(rivelatore 1 E rivelatore 2 click)
    # Per fotoni identici (δφ=0): le fasi si sommano → uscita 1 solo
    # Per fotoni distinguibili (δφ grande): distribuzione uniforme
    
    # Modello semplificato: P_coinc = (1 − V·cos²(δφ))/2
    # dove V è la visibilità
    kappa = beta * abs(c2)
    V = i1(kappa)/i0(kappa) if kappa > 0.001 else kappa/2
    
    P_coinc = 0.5 * (1 - V * np.cos(delta_phi)**2)
    return P_coinc

print("  Modello HOM (2 modi):")
print(f"  {'δt':>6} {'P_coinc(mezzo)':>14} {'P_coinc(QM)':>12} {'P_coinc(class)':>14}")
print(f"  {'─'*50}")

for dt in [0, 0.2, 0.5, 1.0, 1.5, np.pi/2]:
    P_m = hom_two_modes(dt, beta0)
    P_qm = 0.5 * (1 - np.exp(-dt**2))  # Gaussian dip model
    P_cl = 0.5
    print(f"  {dt:6.2f} {P_m:14.4f} {P_qm:12.4f} {P_cl:14.4f}")

print(f"""
  ★ Il modello produce un DIP nella coincidenza (P < 0.5 a δt=0),
  ma la forma è diversa dalla QM. Il dip del mezzo è 
  P_min = {hom_two_modes(0, beta0):.4f} (QM: 0, classico: 0.5).
  
  Con β → ∞ (coerenza perfetta): P_min → 0 (recupera il bunching QM).
  Con β → 0 (decoerente): P_min → 0.5 (recupera il classico).
""")

# ═══════════════════════════════════════════════════════════════════════
# VERDETTO FINALE
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 78}")
print("  VERDETTO FINALE DEI TRE STRESS TEST")
print("═" * 78)

print(f"""
  ┌──────────────────────────────────────────────────┬──────┬─────────────┐
  │  Test                                            │ Pass │   Nota      │
  ├──────────────────────────────────────────────────┼──────┼─────────────┤
  │  I. ESISTENZA FISICA DEL MEZZO                   │      │             │
  │    Stabilità termica (C_v ≥ 0)                   │  ✓   │ esatto      │
  │    Fluttuazione-dissipazione                     │  ✓   │ rapporto 1.0│
  │    Equazione di stato                            │  ✓   │ consistente │
  │    Velocità del suono                            │  ✓   │ finita      │
  │    Osservazione diretta                          │  ✗   │ non possib. │
  │                                                  │      │             │
  │  II. FENOMENOLOGIA MULTIPARTITA                  │      │             │
  │    Correlazioni a 3 corpi (prodotto triplo)      │  ✗   │ rompe NS    │
  │    Correlazioni a coppie (modi indipendenti)     │  ✓   │ OK ma no GHZ│
  │    Disuguaglianza di Mermin (M₃ = 4)            │  ✗   │ max ~0.5    │
  │    Stato W                                       │  ✗   │ non definib.│
  │    Accoppiamento inter-modo (per GHZ)            │  ?   │ da costruire│
  │                                                  │      │             │
  │  III. STATISTICHE QUANTISTICHE                   │      │             │
  │    Interferenza (frange)                          │  ~   │ forma div.  │
  │    Visibilità frange                             │  ~   │ V < V_QM    │
  │    Bunching/Anti-bunching                        │  ✗   │ non emerge  │
  │    HOM dip                                       │  ~   │ parziale    │
  │    Pauli (fermioni)                              │  ✗   │ non emerge  │
  │    Bose-Einstein (bosoni)                        │  ✗   │ non emerge  │
  └──────────────────────────────────────────────────┴──────┴─────────────┘
  
  ═══════════════════════════════════════════════════════════════════════
  DIAGNOSI STRUTTURALE:
  
  I LIMITI NON SONO BUG — SONO LA FRONTIERA DEL MODELLO 1D.
  
  Il framework con un singolo modo su S¹ cattura:
  ✓ Correlazioni bipartite (singoletto, tripletto)
  ✓ No-signaling (per armoniche pari)
  ✓ Termodinamica del mezzo
  ✓ Interferenza qualitativa
  
  NON cattura (e non PUÒ catturare con 1 modo):
  ✗ Entanglement multipartita genuino (serve accoppiamento inter-modo)
  ✗ Statistiche Bose/Fermi (serve antisimmetrizzazione su SPAZIO DI FOCK)
  ✗ Effetto HOM completo (serve struttura modale del beamsplitter)
  ✗ Principio di Pauli (non c'è meccanismo di esclusione)
  
  LA STRADA AVANTI:
  
  Per superare questi limiti, il framework dovrebbe essere ESTESO a:
  
  1) MODI MULTIPLI: N modi φ₁,...,φ_N su (S¹)^N
     → correlazioni multipartite
     → accoppiamento inter-modo → GHZ
  
  2) SIMMETRIZZAZIONE: regola per bosoni/fermioni su (S¹)^N
     → statistiche quantistiche
     → principio di Pauli come vincolo topologico
  
  3) SPAZIO DI FOCK: numero variabile di modi
     → creazione/distruzione di particelle
     → HOM completo
  
  Queste estensioni NON sono banali. Ciascuna richiede nuovo 
  lavoro matematico e nuove dimostrazioni di consistenza.
  
  Ma il punto è: i limiti sono IDENTIFICATI e la strada è CHIARA.
  Non è un fallimento — è una mappa di ciò che resta da costruire.
  ═══════════════════════════════════════════════════════════════════════
""")

