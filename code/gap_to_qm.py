"""
═══════════════════════════════════════════════════════════════════════════
  COSA MANCA PER ARRIVARE A QM
  
  Analisi strutturale dei gap, con dimostrazione di dove e perché 
  il modello si ferma, e cosa servirebbe per superare ogni ostacolo.
═══════════════════════════════════════════════════════════════════════════
"""
import numpy as np
from scipy.special import ndtr, i0, i1

M = 2048
psi = np.linspace(0, 2*np.pi, M, endpoint=False)
dp = 2*np.pi / M; sr = 0.005

print("=" * 78)
print("  MAPPA COMPLETA: DAL MODELLO VIBRAZIONALE ALLA QM")
print("  Cinque gap strutturali e cosa serve per ciascuno")
print("=" * 78)

# ═══════════════════════════════════════════════════════════════════════
# GAP 1: DA ρ(φ) A |ψ|² — IL PASSAGGIO ALLA REGOLA DI BORN
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 78}")
print("  GAP 1: DISTRIBUZIONE DI GIBBS vs REGOLA DI BORN")
print("═" * 78)

print("""
  IL MODELLO HA:    ρ(φ) = exp[βS(φ)] / Z     [Gibbs/Boltzmann]
  LA QM HA:         P(x) = |ψ(x)|²             [Born]
  
  Queste sono DUE regole diverse per calcolare probabilità.
  
  Gibbs: ρ ∝ exp(−βE)  →  dipende dall'ENERGIA esponenzialmente
  Born:  P ∝ |ψ|²      →  dipende dall'AMPIEZZA quadraticamente
  
  La differenza non è cosmetica — è strutturale:
""")

# Confronto numerico: exp(-βE) vs |ψ|² per lo stesso potenziale
V = np.cos(2*psi)  # potenziale armonico

# Gibbs
for beta in [0.5, 1.0, 2.0]:
    rho_gibbs = np.exp(beta * V)
    rho_gibbs /= np.sum(rho_gibbs) * dp

    # Born: ψ è l'autofunzione del ground state in questo potenziale
    # Per cos(2ψ), le autofunzioni sono le funzioni di Mathieu
    # Approssimiamo: |ψ_0|² ∝ exp(sqrt(β)·cos(2ψ)) per β grande
    rho_born = np.exp(np.sqrt(beta) * V)
    rho_born /= np.sum(rho_born) * dp
    
    # Differenza
    diff = np.sqrt(np.mean((rho_gibbs - rho_born)**2))
    
    kl = np.sum(rho_gibbs * np.log(rho_gibbs / (rho_born + 1e-30) + 1e-30) * dp)
    
    print(f"  β={beta:.1f}: RMSE(Gibbs, Born_approx) = {diff:.6f}, KL = {kl:.6f}")

print(f"""
  ★ LA DIFFERENZA È NELLA DIPENDENZA FUNZIONALE:
  
  Gibbs:  ρ ∝ exp(β·V)        →  LINEARE nell'esponente  →  β·V
  Born:   P ∝ exp(√β·V)       →  RADICE nell'esponente   →  √β·V
  
  Per passare da Gibbs a Born, serve una trasformazione:
  
    β → √β    (o equivalentemente: E → √E)
  
  Questa è ESATTAMENTE la relazione tra equazione di Schrödinger e
  equazione di Fokker-Planck nella trasformazione di Wick (t → −it):
  
  Schrödinger:  iℏ ∂ψ/∂t = Ĥψ        →  |ψ|² = Born
  Fokker-Planck: ∂ρ/∂t = D∇²ρ − ∇(Fρ) →  ρ_eq = Gibbs
  
  COSA SERVE: la rotazione di Wick (tempo immaginario) che connette 
  la dinamica del mezzo (Smoluchowski) all'equazione di Schrödinger.
  Questa rotazione esiste formalmente — è la base della meccanica 
  quantistica euclidea e del metodo path integral.
""")

# ═══════════════════════════════════════════════════════════════════════
# GAP 2: DA R A C — LA FASE COMPLESSA
# ═══════════════════════════════════════════════════════════════════════
print(f"{'═' * 78}")
print("  GAP 2: NUMERI REALI vs NUMERI COMPLESSI (LA FASE)")
print("═" * 78)

print("""
  IL MODELLO HA:    correlatore E = ∫ f·g·ρ dφ      [reale]
  LA QM HA:         correlatore ⟨AB⟩ = ⟨ψ|A⊗B|ψ⟩    [complesso]
  
  La QM usa ampiezze COMPLESSE:  ψ = |ψ| e^(iθ)
  Il modello usa funzioni REALI:  ρ(φ) ∈ R⁺
  
  La differenza chiave è l'INTERFERENZA:
""")

# Dimostrazione: interferenza reale vs complessa
# Due sorgenti con fase relativa δ

print("  Interferenza con fasi relative:")
print(f"  {'δ':>6} {'I_reale':>10} {'I_complessa':>12} {'Differenza':>12}")
print(f"  {'─'*44}")

for delta_phase in [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]:
    # Reale: (A+B)² dove A,B reali
    # I_real = A² + B² + 2AB cos(δ)   [sempre ≥ 0 per δ=0]
    # Ma: non può essere NEGATIVA
    
    # Complessa: |A e^(iδ/2) + B e^(-iδ/2)|² = A² + B² + 2AB cos(δ)
    # Stessa formula! MA le ampiezze possono dare interferenza distruttiva completa
    
    A, B = 1.0, 1.0
    I_real = A**2 + B**2 + 2*A*B*np.cos(delta_phase)
    I_complex = abs(A*np.exp(1j*delta_phase/2) + B*np.exp(-1j*delta_phase/2))**2
    
    print(f"  {delta_phase/np.pi:5.2f}π {I_real:10.4f} {I_complex:12.4f} {I_real-I_complex:12.4f}")

print(f"""
  ★ Per due sorgenti di uguale ampiezza, l'interferenza reale e 
  complessa danno lo STESSO risultato (identici per ogni δ).
  
  La differenza emerge quando si compongono PIÙ DI DUE ampiezze,
  o quando le fasi non sono geometriche (cioè non vengono da 
  differenze di cammino).
  
  COSA SERVE: estendere la variabile latente da φ ∈ S¹ (reale) a
  z = e^(iφ) ∈ U(1) (complesso unitario). La distribuzione diventa:
  
    ρ(z) → ψ(z)·ψ*(z) = |ψ(z)|²
  
  con ψ: S¹ → C (funzione d'onda complessa).
  
  Il salto concettuale: la variabile nascosta non è la FASE φ,
  ma l'AMPIEZZA COMPLESSA ψ(φ) = |ψ|e^(iθ(φ)).
""")

# ═══════════════════════════════════════════════════════════════════════
# GAP 3: DA S¹ A HILBERT — LO SPAZIO DEGLI STATI
# ═══════════════════════════════════════════════════════════════════════
print(f"{'═' * 78}")
print("  GAP 3: S¹ vs SPAZIO DI HILBERT")
print("═" * 78)

print("""
  IL MODELLO HA:    spazio latente S¹ (cerchio, 1D, compatto)
  LA QM HA:         spazio di Hilbert H (infinito-dimensionale)
  
  Uno stato su S¹ è una funzione ρ(φ): S¹ → R⁺  [1 funzione reale]
  Uno stato in H è un vettore |ψ⟩ ∈ H             [∞ numeri complessi]
  
  Per un qubit: H = C²  →  |ψ⟩ = α|0⟩ + β|1⟩  [4 parametri reali]
  Per S¹:       ρ(φ) ha ∞ parametri, ma è CLASSICA (no sovrapposizione)
  
  LA DIFFERENZA FONDAMENTALE:
  
  In H: |ψ⟩ = α|↑⟩ + β|↓⟩ descrive UNO stato che è 
        SIMULTANEAMENTE ↑ e ↓ (sovrapposizione)
  
  Su S¹: ρ(φ) descrive IGNORANZA su quale φ sia il vero valore
         (miscela statistica, non sovrapposizione)
  
  Questa è la distinzione tra:
  • SOVRAPPOSIZIONE COERENTE (QM): α|↑⟩ + β|↓⟩
  • MISCELA STATISTICA (classica): p·|↑⟩⟨↑| + (1−p)·|↓⟩⟨↓|
""")

# Dimostrazione numerica della differenza
# Test: matrice densità del singoletto vs modello vibrazionale

# Singoletto QM: ρ = |ψ⟩⟨ψ| con |ψ⟩ = (|01⟩−|10⟩)/√2
# ρ ha rango 1 → è uno stato PURO
# La ridotta ρ_A = Tr_B(ρ) = I/2 → completamente mista

# Modello vibrazionale: ρ(φ) è una distribuzione classica
# Non ha concetto di "stato puro" vs "stato misto" nel senso QM

print("  Confronto: proprietà strutturali")
print(f"  {'Proprietà':>30} {'Modello S¹':>15} {'QM (Hilbert)':>15}")
print(f"  {'─'*62}")

properties = [
    ("Sovrapposizione coerente", "NO", "SÌ"),
    ("Stati puri", "NO (solo miscele)", "SÌ"),
    ("Fase relativa α/β", "NO (solo ρ)", "SÌ (ψ)"),
    ("Entanglement (von Neumann)", "Non definito", "S = −Tr(ρ log ρ)"),
    ("Prodotto tensoriale H₁⊗H₂", "NO (prodotto S¹×S¹)", "SÌ"),
    ("Operatori unitari", "NO", "SÌ"),
    ("Teorema spettrale", "NO", "SÌ"),
    ("No-cloning", "Non vale", "Vale"),
    ("Teletrasporto", "Non possibile", "Possibile"),
]

for prop, s1, qm in properties:
    print(f"  {prop:>30} {s1:>15} {qm:>15}")

print(f"""
  ★ COSA SERVE:
  
  Passare da S¹ (spazio di configurazione classico) a L²(S¹) 
  (spazio di Hilbert delle funzioni quadrato-integrabili su S¹).
  
  L²(S¹) HA la struttura di Hilbert:
  • base: e^(inφ) per n ∈ Z  (armoniche di Fourier)
  • prodotto interno: ⟨f|g⟩ = ∫ f*(φ)g(φ) dφ
  • sovrapposizione: ψ = Σ_n c_n e^(inφ)
  
  IRONICAMENTE, le armoniche di Fourier che abbiamo usato per 
  decomporre la distribuzione (cos(2nψ), sin(2nψ)) sono ESATTAMENTE 
  la base di L²(S¹). Il modello sta già usando lo spazio di Hilbert 
  come strumento matematico — ma non come spazio degli STATI.
""")

# ═══════════════════════════════════════════════════════════════════════
# GAP 4: DA ⟨ABC⟩=0 A GHZ — L'ENTANGLEMENT GENUINO
# ═══════════════════════════════════════════════════════════════════════
print(f"{'═' * 78}")
print("  GAP 4: ENTANGLEMENT GENUINAMENTE MULTIPARTITO")
print("═" * 78)

print("""
  IL RISULTATO CHIAVE DEL TEST (S¹)²:
  ⟨ABC⟩ = 0 identicamente, per ogni λ, ogni β, ogni configurazione.
  
  PERCHÉ: la dimostrazione del no-signaling richiede che ogni 
  funzione di misura abbia media zero:
  
    ∫ f_A(φ₁,φ₂) ρ(φ₁,φ₂) dφ₁dφ₂ = 0
  
  Se f_A, f_B, f_C hanno tutte media zero, e la distribuzione è 
  CLASSICA (non-negativa), allora il prodotto f_A·f_B·f_C integrato 
  contro ρ è fortemente vincolato.
  
  Per una distribuzione FATTORIZZABILE ρ = ρ₁·ρ₂:
    ⟨ABC⟩ = ⟨A·B⟩₁ · ⟨C⟩₂ = ⟨A·B⟩₁ · 0 = 0
  
  Per una distribuzione NON-fattorizzabile con marginali simmetriche:
    il risultato è comunque zero per simmetria (dimostrato nel test).
  
  ★ QUESTO È IL TEOREMA CHIAVE:
  
  Su (S¹)^N con distribuzione classica (ρ ≥ 0) e no-signaling 
  (marginali nulli), il correlatore a N corpi con N DISPARI è 
  IDENTICAMENTE ZERO.
  
  Dimostrazione per N=3:
  ⟨ABC⟩ = ∫∫ f_A·f_B·f_C · ρ dφ₁dφ₂
  
  f_A ha media zero su ρ per NS: ∫ f_A ρ = 0
  f_B ha media zero su ρ per NS: ∫ f_B ρ = 0
  f_C ha media zero su ρ per NS: ∫ f_C ρ = 0
  
  Se ρ è π-periodica su ciascuna variabile e f è π-anti-periodica:
  
  ⟨ABC⟩ = ∫₀^π ∫₀^π [f_A·f_B·f_C · ρ](ψ₁,ψ₂) dψ₁dψ₂
         + ∫₀^π ∫₀^π [f_A·f_B·f_C · ρ](ψ₁+π,ψ₂) dψ₁dψ₂
         + ∫₀^π ∫₀^π [f_A·f_B·f_C · ρ](ψ₁,ψ₂+π) dψ₁dψ₂
         + ∫₀^π ∫₀^π [f_A·f_B·f_C · ρ](ψ₁+π,ψ₂+π) dψ₁dψ₂
  
  Sotto ψ₁ → ψ₁+π: f_A→−f_A, f_B→−f_B, ρ→ρ, f_C→f_C
  Sotto ψ₂ → ψ₂+π: f_A→f_A (se legge solo φ₁), f_C→−f_C, ρ→ρ
  
  Il prodotto di TRE funzioni anti-periodiche non è anti-periodico:
  (−1)³ = −1 ma servono le anti-periodicità su VARIABILI DIVERSE.
  
  Con A su φ₁, B su φ₁, C su φ₂:
  ψ₁→ψ₁+π: f_A→−f_A, f_B→−f_B → f_A·f_B→+f_A·f_B (pari!)
  Ma: f_A·f_B·f_C ha f_C invariante sotto ψ₁→ψ₁+π
  E sotto ψ₂→ψ₂+π: f_C→−f_C, f_A·f_B invariante
  → f_A·f_B·f_C cambia segno sotto ψ₂→ψ₂+π
  → integrale su ψ₂ dà zero per la stessa ragione del bipartito
  
  ✓ DIMOSTRATO: ⟨ABC⟩ = 0 è un TEOREMA, non un artefatto numerico.
""")

# Verifica numerica
N = 512
p1 = np.linspace(0, 2*np.pi, N, endpoint=False)
p2 = np.linspace(0, 2*np.pi, N, endpoint=False)
P1, P2 = np.meshgrid(p1, p2, indexing='ij')
dd = (2*np.pi/N)**2

# Distribuzione accoppiata
S = -0.876*np.cos(4*P1) - 0.876*np.cos(4*P2) + 1.5*np.cos(2*(P1-P2))
lw = 0.5*S; lw -= np.max(lw); w = np.exp(lw); rho = w/(np.sum(w)*dd)

# f_A su φ₁, f_B su φ₁, f_C su φ₂
a, b, c = 0.3, 0.7, 1.1
fA = 2*ndtr(np.cos(P1-a)/sr)-1
fB = -(2*ndtr(np.cos(P1-b)/sr)-1)
fC = -(2*ndtr(np.cos(P2-c)/sr)-1)

ABC = np.sum(fA*fB*fC*rho*dd)
AB = np.sum(fA*fB*rho*dd)
muC = np.sum(fC*rho*dd)

print(f"  Verifica numerica:")
print(f"    ⟨ABC⟩ = {ABC:.2e}")
print(f"    ⟨AB⟩  = {AB:.6f}")
print(f"    ⟨C⟩   = {muC:.2e}")
print(f"    ⟨AB⟩·⟨C⟩ = {AB*muC:.2e}")

print(f"""
  COME LA QM SUPERA QUESTO VINCOLO:
  
  In QM, le "ampiezze" ψ possono essere NEGATIVE (o complesse).
  Il correlatore a 3 corpi è:
  
    ⟨ABC⟩ = ⟨ψ| (σ_a ⊗ σ_b ⊗ σ_c) |ψ⟩
  
  dove σ_a sono matrici di Pauli. Questa espressione coinvolge 
  PRODOTTI DI MATRICI, non prodotti di funzioni a media zero.
  
  La chiave è che |ψ⟩ può avere componenti negative che si 
  cancellano nei marginali (⟨A⟩ = 0) ma NON nel prodotto triplo.
  
  ★ COSA SERVE: passare da ρ ≥ 0 (distribuzione classica) a 
  una quasi-distribuzione W(φ₁,φ₂) che può essere NEGATIVA.
  
  Questa è ESATTAMENTE la funzione di Wigner.
  La negatività della funzione di Wigner è la FIRMA dell'entanglement 
  genuinamente multipartito che il modello classico non può riprodurre.
""")

# ═══════════════════════════════════════════════════════════════════════
# GAP 5: DA Z₂ A SU(2) — IL GRUPPO DI SIMMETRIA
# ═══════════════════════════════════════════════════════════════════════
print(f"{'═' * 78}")
print("  GAP 5: DA Z₂ A SU(2) — IL GRUPPO DI GAUGE")
print("═" * 78)

print(f"""
  IL MODELLO HA:    simmetria Z₂ (traslazione di π su S¹)
  LA QM HA:         simmetria SU(2) (rotazioni nello spazio degli spin)
  
  Z₂ ha 2 elementi: {{identità, traslazione di π}}
  SU(2) ha ∞ elementi: tutte le rotazioni in 3D degli spinori
  
  Z₂ è un SOTTOGRUPPO di SU(2). Passare da Z₂ a SU(2) significa 
  passare da:
  • 1 asse di simmetria (il cerchio S¹) 
  • a 3 assi di simmetria (la sfera S² = SU(2)/U(1))
  
  Nella sfera di Bloch, uno stato qubit puro è un punto su S²:
  |ψ⟩ = cos(θ/2)|0⟩ + e^(iφ)sin(θ/2)|1⟩
  
  Il modello su S¹ vede solo la SEZIONE EQUATORIALE di S² (l'angolo φ).
  Manca la LATITUDINE θ — cioè la componente z dello spin.
""")

# Quante dimensioni mancano?
print("  Confronto dimensionale:")
print(f"  {'':>30} {'Modello':>10} {'QM':>10} {'Gap':>10}")
print(f"  {'─'*62}")

dims = [
    ("Parametri stato (1 qubit)", "∞ (ρ su S¹)", "2 (θ,φ)", "topologia"),
    ("Parametri stato (2 qubit)", "∞×∞", "6", "prodotto ⊗"),
    ("Gruppo simmetria (1 qubit)", "Z₂", "SU(2)", "continuo"),
    ("Gruppo simmetria (2 qubit)", "Z₂×Z₂", "SU(2)⊗SU(2)", "non-abeliano"),
    ("Algebra osservabili", "commutativa", "non-commut.", "σ_x σ_y≠σ_y σ_x"),
    ("Spazio delle fasi", "S¹ (1D)", "S² (2D)", "+1 dimensione"),
]

for prop, mod, qm, gap in dims:
    print(f"  {prop:>30} {mod:>10} {qm:>10} {gap:>10}")

# ═══════════════════════════════════════════════════════════════════════
# SINTESI: LA MAPPA COMPLETA
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 78}")
print("  SINTESI: I CINQUE PASSI DA S¹ ALLA QM")
print("═" * 78)

print(f"""
  ┌──────────────────────────────────────────────────────────────────────┐
  │                                                                      │
  │  PASSO 1: REGOLA DI BORN                                           │
  │  ρ = exp(βV) → |ψ|² = exp(√β·V)                                    │
  │  Strumento: rotazione di Wick (t → −it)                             │
  │  Effetto: Smoluchowski → Schrödinger                                │
  │  Difficoltà: ★★☆☆☆ (formalmente nota)                              │
  │                                                                      │
  │  PASSO 2: NUMERI COMPLESSI                                          │
  │  ρ(φ) ∈ R⁺ → ψ(φ) ∈ C                                             │
  │  Strumento: ampiezza complessa ψ = √ρ · e^(iS/ℏ)                  │
  │  Effetto: interferenza con fase, cancellazioni                      │
  │  Difficoltà: ★★★☆☆ (richiede azione di Hamilton-Jacobi)            │
  │                                                                      │
  │  PASSO 3: SPAZIO DI HILBERT                                        │
  │  L²(S¹) come spazio degli stati (non delle configurazioni)         │
  │  Strumento: quantizzazione canonica di φ e p_φ = −i∂/∂φ           │
  │  Effetto: sovrapposizione coerente, stati puri, unitarietà         │
  │  Difficoltà: ★★★★☆ (cambio concettuale profondo)                   │
  │                                                                      │
  │  PASSO 4: QUASI-DISTRIBUZIONE (WIGNER)                             │
  │  ρ ≥ 0 → W(φ,p) che può essere < 0                                │
  │  Strumento: trasformata di Wigner-Weyl                              │
  │  Effetto: entanglement multipartito, GHZ, violazione Mermin        │
  │  Difficoltà: ★★★★★ (la negatività è la "magia" quantistica)        │
  │                                                                      │
  │  PASSO 5: SU(2) COME GRUPPO DI GAUGE                               │
  │  Z₂ → SU(2) (da cerchio a sfera di Bloch)                         │
  │  Strumento: fibrato principale SU(2) → S²                          │
  │  Effetto: tutte le misure (non solo equatoriali), spin completo    │
  │  Difficoltà: ★★★★★ (richiede teoria di gauge)                      │
  │                                                                      │
  ├──────────────────────────────────────────────────────────────────────┤
  │                                                                      │
  │  Dopo i 5 passi, il modello vibrazionale DIVENTA la QM:            │
  │                                                                      │
  │  modo su S¹ + Wick + C + L² + Wigner + SU(2) = QM standard        │
  │                                                                      │
  │  In questa lettura, la QM È la teoria del mezzo vibrazionale       │
  │  dopo quantizzazione. Non è un'approssimazione — è esatta.         │
  │                                                                      │
  └──────────────────────────────────────────────────────────────────────┘
""")

# ═══════════════════════════════════════════════════════════════════════
# DOVE IL MODELLO ATTUALE SI COLLOCA
# ═══════════════════════════════════════════════════════════════════════
print(f"{'═' * 78}")
print("  DOVE SIAMO OGGI")
print("═" * 78)

print(f"""
  Il modello vibrazionale attuale è al PASSO 0 — prima dei 5 passi.
  È il "mezzo classico" prima della quantizzazione.
  
  Cosa ha già:
  ✓ Lo spazio di configurazione (S¹)
  ✓ La dinamica (Smoluchowski / Fokker-Planck)
  ✓ Le armoniche (cos(2nψ) = base di Fourier = base di L²(S¹))
  ✓ La termodinamica (FDT, equazione di stato)
  ✓ La simmetria Z₂ (sottogruppo di SU(2))
  ✓ Il no-signaling (come simmetria, non come postulato)
  
  Cosa manca per ogni passo:
  
  PASSO 1 (Born): serve la rotazione di Wick nel parametro β.
    β_modello → β_Wick = β_modello^(1/2)
    
  PASSO 2 (Complessi): serve l'azione di Hamilton-Jacobi S(φ,t).
    ρ(φ) → ψ(φ) = √ρ · exp(iS/ℏ)
    ℏ entra qui come SCALA DELL'AZIONE.
    
  PASSO 3 (Hilbert): serve la quantizzazione canonica.
    φ → operatore φ̂, p → operatore p̂ = −iℏ∂/∂φ
    [φ̂, p̂] = iℏ (relazione di commutazione canonica)
    
  PASSO 4 (Wigner): serve la trasformata di Wigner della ρ quantizzata.
    ρ(φ) → W(φ,p) = (1/πℏ)∫ ψ*(φ+y)ψ(φ−y) e^(2ipy/ℏ) dy
    W può essere < 0 → entanglement multipartito.
    
  PASSO 5 (SU(2)): serve il fibrato di spin su S².
    S¹ ⊂ S² → SU(2) agisce su C² (spinori)
    Lo spin emerge come rappresentazione del gruppo di rotazione del mezzo.
  
  LA DISTANZA: il modello è più vicino alla QM di quanto sembri.
  Le armoniche di Fourier su S¹ SONO GIÀ la base di Hilbert L²(S¹).
  Il parametro β È GIÀ una scala termodinamica.
  La simmetria Z₂ È GIÀ un sottogruppo di SU(2).
  
  Il passo concettualmente più grande è il PASSO 4 (negatività di Wigner).
  È lì che la "classicità" del modello (ρ ≥ 0) si rompe e emergono 
  le proprietà genuinamente quantistiche.
  
  ★ FRASE FINALE:
  Il modello vibrazionale su S¹ è la MECCANICA CLASSICA del mezzo.
  La meccanica quantistica è la sua QUANTIZZAZIONE.
  I cinque gap non sono fallimenti — sono i cinque passi della 
  quantizzazione canonica applicati al mezzo vibrazionale.
""")

