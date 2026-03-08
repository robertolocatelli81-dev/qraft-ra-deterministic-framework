"""
═══════════════════════════════════════════════════════════════════════════
  CONNESSIONE DI FOURIER TRA T4 E BELL
  E PROVE SPERIMENTALI SULLA "PROFONDITÀ DI MEMORIA" DEL MEZZO
  
  Idea chiave: T4 e Bell sono entrambi vincoli sul settore armonico.
  La "profondità di memoria" è l'informazione multi-tempo che 
  separa il classico dal quantistico — e potrebbe essere misurabile.
═══════════════════════════════════════════════════════════════════════════
"""
import numpy as np
from scipy.special import ndtr, i0, i1
from scipy.linalg import eigh

M = 1024
phi = np.linspace(0, 2*np.pi, M, endpoint=False)
dp = 2*np.pi / M; sr = 0.005
c2 = -0.876; beta = 0.779; h2_2m = 0.154
kappa = beta*abs(c2)

print("=" * 78)
print("  CONNESSIONE DI FOURIER T4 ↔ BELL")
print("  E PROFONDITÀ DI MEMORIA DEL MEZZO")
print("=" * 78)

# ═══════════════════════════════════════════════════════════════════════
# 1. LA STRUTTURA PARALLELA: T4 E BELL COME VINCOLI ARMONICI
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'━' * 78}")
print("  1. T4 E BELL: DUE FACCE DELLO STESSO VINCOLO")
print("━" * 78)

print("""
  BELL (1964):
  ρ(λ) ≥ 0  +  località  →  |S_CHSH| ≤ 2
  
  In forma armonica: il correlatore classico E(Δθ) = ∫f·g·ρ dφ 
  è vincolato dal fatto che ρ ≥ 0 impone BOUND sulle ampiezze 
  delle armoniche di Fourier di E(Δθ).
  
  T4 (nostro):
  ρ(φ) ≥ 0  +  f π-anti-periodica  →  ∫(A·B)·ψ₀*ψ₁ dφ = 0
  
  In forma armonica: A·B è confinata nel settore pari di Fourier,
  ψ₀*ψ₁ ha componenti dispari → prodotto interno = 0.
  
  LA CONNESSIONE:
  Entrambi sono vincoli sul CONTENUTO ARMONICO delle correlazioni 
  imposti dalla positività di ρ. Bell limita l'AMPIEZZA totale;
  T4 limita i SETTORI accessibili.
""")

# Dimostrazione quantitativa: decomposizione di Fourier di E(Δθ)
rho = np.exp(kappa*np.cos(4*phi))/(2*np.pi*i0(kappa))

# Correlatore classico come funzione di Δθ
E_classical = []
thetas = np.linspace(0, np.pi, 100)
for theta in thetas:
    d = theta/2
    A = 2*ndtr(np.cos(phi+d)/sr)-1
    B = -(2*ndtr(np.cos(phi-d)/sr)-1)
    E_classical.append(np.sum(A*B*rho*dp))
E_classical = np.array(E_classical)

# Decomposizione di Fourier di E_cl(Δθ)
print("  Armoniche di Fourier di E_classico(Δθ):")
print(f"  {'k':>4} {'a_k (cos)':>12} {'b_k (sin)':>12} {'|c_k|':>10} {'cumul %':>10}")
print(f"  {'─'*48}")

total_power = np.sum(E_classical**2) * (thetas[1]-thetas[0])
cumul = 0
fourier_coeffs = []
for k in range(8):
    ak = 2/np.pi * np.trapezoid(E_classical*np.cos(k*thetas), thetas)
    bk = 2/np.pi * np.trapezoid(E_classical*np.sin(k*thetas), thetas)
    ck = np.sqrt(ak**2 + bk**2)
    power_k = (ak**2 + bk**2) * np.pi / 2
    cumul += power_k
    fourier_coeffs.append((k, ak, bk, ck))
    pct = cumul/total_power*100 if total_power > 0 else 0
    print(f"  {k:4d} {ak:12.6f} {bk:12.6f} {ck:10.6f} {pct:9.1f}%")

# Confronto: QM ha solo cos(Δθ) → k=1 puro
print(f"""
  ★ BELL IN FORMA ARMONICA:
  
  Il correlatore QM è E = −cos(Δθ) → pura armonica k=1.
  Il correlatore classico contiene k=1, 3, 5, 7 (tutte dispari).
  
  La potenza nelle armoniche superiori (k≥3) è la MISURA 
  della deviazione dal singoletto. Bell dice che questa 
  deviazione produce CHSH ≤ 2; il nostro modello la quantifica.
  
  Il RESIDUO armonico (k≥3) è il "costo" della classicità:
""")

# Calcolo esatto del residuo armonico
power_k1 = fourier_coeffs[1][3]**2 * np.pi/2
residual_power = total_power - power_k1
print(f"  Potenza in k=1: {power_k1/total_power*100:.1f}%")
print(f"  Potenza residua (k≥3): {residual_power/total_power*100:.1f}%")

# ═══════════════════════════════════════════════════════════════════════
# 2. IL VINCOLO DI FOURIER COME LIMITE DI MEMORIA
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'━' * 78}")
print("  2. IL VINCOLO DI FOURIER = LIMITE DI MEMORIA")
print("━" * 78)

print("""
  La distribuzione classica ρ(φ) contiene SOLO informazione 
  a tempo singolo — è uno snapshot istantaneo del mezzo.
  
  In un processo stocastico, l'informazione completa include 
  le correlazioni MULTI-TEMPO:
  
  Ordine 1: ⟨X(t)⟩               → media (= 0 per NS)
  Ordine 2: ⟨X(t)X(t+τ)⟩         → correlazione a 2 tempi
  Ordine 3: ⟨X(t)X(t+τ₁)X(t+τ₂)⟩ → correlazione a 3 tempi
  ...
  Ordine n: ⟨X(t₁)...X(t_n)⟩     → correlazione a n tempi
  
  Definiamo la "profondità di memoria" d_m come l'ordine minimo 
  di correlazione multi-tempo necessario per catturare tutta 
  l'informazione del processo.
  
  Per un processo MARKOVIANO: d_m = 2 (basta C(τ))
  Per un processo NON-MARKOVIANO: d_m > 2
  Per la QM completa: d_m = ∞ (tutte le correlazioni servono)
""")

# Simulazione Langevin per estrarre le correlazioni multi-tempo
def langevin_correlations(V_func, dV_func, beta, D, max_order=4,
                           n_traj=500, n_steps=30000, n_equil=5000, dt=0.002):
    phi_arr = np.random.uniform(0, 2*np.pi, n_traj)
    
    for _ in range(n_equil):
        noise = np.sqrt(2*D*dt)*np.random.randn(n_traj)
        phi_arr = (phi_arr - beta*dV_func(phi_arr)*dt + noise) % (2*np.pi)
    
    # Osservabile: X = cos(4φ) (modo quadrupolare)
    X_history = np.zeros((n_steps, n_traj))
    for t in range(n_steps):
        noise = np.sqrt(2*D*dt)*np.random.randn(n_traj)
        phi_arr = (phi_arr - beta*dV_func(phi_arr)*dt + noise) % (2*np.pi)
        X_history[t] = np.cos(4*phi_arr)
    
    return X_history

V = lambda p: c2*np.cos(4*p)
dV = lambda p: -4*c2*np.sin(4*p)
D = 1/(2*beta)

print("  Simulazione Langevin per correlazioni multi-tempo...")
X = langevin_correlations(V, dV, beta, D, n_traj=300, n_steps=20000)

mean_X = np.mean(X)
X_c = X - mean_X  # centrata
sigma = np.std(X_c)

# Correlazione a 2 tempi normalizzata
def C2(tau):
    if tau == 0: return 1.0
    return np.mean(X_c[tau:]*X_c[:-tau]) / sigma**2

# Cumulante a 3 tempi normalizzato
def C3(t1, t2):
    n = len(X_c) - t1 - t2
    if n < 100: return 0
    return np.mean(X_c[:n]*X_c[t1:n+t1]*X_c[t1+t2:n+t1+t2]) / sigma**3

# Cumulante a 4 tempi (connesso)
def C4(t1, t2, t3):
    n = len(X_c) - t1 - t2 - t3
    if n < 100: return 0
    raw = np.mean(X_c[:n]*X_c[t1:n+t1]*X_c[t1+t2:n+t1+t2]*X_c[t1+t2+t3:n+t1+t2+t3])
    # Sottrai prodotti di C2 (formula dei cumulanti connessi)
    c4_conn = raw/sigma**4 - C2(t1)*C2(t3) - C2(t2)*C2(t1+t2+t3) - C2(t1+t2)*C2(t2+t3)
    return c4_conn

# ═══════════════════════════════════════════════════════════════════════
# 3. PROFILO DI MEMORIA: CUMULANTI A ORDINI CRESCENTI
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'━' * 78}")
print("  3. PROFILO DI MEMORIA DEL MEZZO VIBRAZIONALE")
print("━" * 78)

# Correlazione a 2 tempi
print("  Ordine 2: C₂(τ)")
taus = [0, 10, 50, 100, 500, 1000, 2000, 5000]
print(f"  {'τ':>6} {'C₂(τ)':>10} {'|C₂|':>10}")
print(f"  {'─'*28}")
for tau in taus:
    c = C2(tau)
    print(f"  {tau:6d} {c:10.4f} {abs(c):10.4f}")

# Tempo di correlazione
tau_c = 0
for t in range(1, 5000):
    if abs(C2(t)) < 1/np.e:
        tau_c = t; break
print(f"  τ_c (ordine 2) = {tau_c}")

# Cumulante a 3 tempi: NON-GAUSSIANITÀ = MEMORIA OLTRE ORDINE 2
print(f"\n  Ordine 3: κ₃(τ₁,τ₂) / σ³ (non-Gaussianità)")
print(f"  {'τ₁':>6} {'τ₂':>6} {'κ₃/σ³':>10} {'|κ₃|/σ³':>10}")
print(f"  {'─'*36}")

max_c3 = 0
for t1 in [10, 50, 100, 500, 1000]:
    for t2 in [10, 50, 100, 500]:
        c3 = C3(t1, t2)
        max_c3 = max(max_c3, abs(c3))
        if t2 in [10, 100, 500]:
            print(f"  {t1:6d} {t2:6d} {c3:+10.4f} {abs(c3):10.4f}")

print(f"  max|κ₃|/σ³ = {max_c3:.4f}")

# Cumulante a 4 tempi: MEMORIA ANCORA PIÙ PROFONDA
print(f"\n  Ordine 4: κ₄(τ₁,τ₂,τ₃) connesso (memoria profonda)")
print(f"  {'(τ₁,τ₂,τ₃)':>20} {'κ₄_conn':>12}")
print(f"  {'─'*36}")

max_c4 = 0
for t1,t2,t3 in [(10,10,10),(50,50,50),(100,100,100),(10,50,100),(50,100,500)]:
    c4 = C4(t1,t2,t3)
    max_c4 = max(max_c4, abs(c4))
    print(f"  ({t1},{t2},{t3}){' '*(16-len(f'({t1},{t2},{t3})'))} {c4:+12.4f}")

print(f"  max|κ₄_conn| = {max_c4:.4f}")

print(f"""
  ★ PROFONDITÀ DI MEMORIA DEL MEZZO:
  
  Ordine 2 (C₂): dominante, τ_c = {tau_c} passi
  Ordine 3 (κ₃): max = {max_c3:.4f} {'(significativo)' if max_c3 > 0.1 else '(debole)'}
  Ordine 4 (κ₄): max = {max_c4:.4f} {'(significativo)' if max_c4 > 0.1 else '(debole)'}
  
  La profondità di memoria effettiva è d_m ≈ {'2 (quasi-Gaussiano)' if max_c3 < 0.1 else '3+ (non-Gaussiano)'}
""")

# ═══════════════════════════════════════════════════════════════════════
# 4. LA CONNESSIONE: MEMORIA ↔ SETTORE ARMONICO ↔ BELL
# ═══════════════════════════════════════════════════════════════════════
print(f"{'━' * 78}")
print("  4. COME LA MEMORIA SI CONNETTE AL BOUND DI BELL")
print("━" * 78)

print("""
  La catena logica è:
  
  (1) PROFONDITÀ DI MEMORIA d_m determina quanta informazione 
      del processo stocastico è accessibile.
  
  (2) Con d_m = 2 (Markoviano): solo C₂(τ) disponibile.
      Le correlazioni bipartite sono limitate dal contenuto 
      armonico di C₂ → vincolo tipo Bell.
  
  (3) Con d_m > 2 (non-Markoviano): κ₃, κ₄, ... disponibili.
      Informazione aggiuntiva → possibile accedere a settori 
      armonici superiori → possibile superare Bell?
  
  (4) Con d_m = ∞ (QM completa): tutte le correlazioni.
      Nessun vincolo armonico → CHSH fino a 2√2.
  
  TEST: la violazione di Bell CRESCE con la profondità di memoria?
""")

# Simuliamo: correlatore "arricchito" che usa informazione multi-tempo
def enriched_correlator(X_history, theta_a, theta_b, order=2):
    """
    Correlatore che usa informazione a ordini crescenti.
    
    Ordine 2: usa solo la distribuzione stazionaria (classico)
    Ordine 3: aggiunge correzione dal cumulante a 3 tempi
    Ordine 4: aggiunge correzione dal cumulante a 4 tempi
    """
    n_steps, n_traj = X_history.shape
    
    # Mappa da X(t) = cos(4φ(t)) all'osservabile di misura
    # A(φ) proiettata su X: A_eff(X) ≈ sign(cos(θ_a)) * X (linearizzazione)
    # Questo è grezzo ma cattura l'idea
    
    # Ordine 2: correlazione stazionaria standard
    da = theta_a/2; db = theta_b/2
    
    # Ricostruiamo φ da X = cos(4φ) → φ = arccos(X)/4
    # Problema: non invertibile. Usiamo direttamente le traiettorie φ
    # Ma non abbiamo φ, solo X = cos(4φ)
    
    # Alternativa: usiamo le correlazioni nel tempo per migliorare la stima
    
    # Per ordine 2: E ≈ ⟨A(t)B(t)⟩ (tempo singolo)
    # Per ordine 3: E ≈ ⟨A(t)B(t)⟩ + α₃ · ⟨A(t)B(t)X(t+τ)⟩ 
    
    # Ma questo richiede le traiettorie φ, non solo X
    # Ritorniamo al modello analitico
    
    return None  # placeholder

# Approccio analitico: come la memoria modifica il bound
print("  Approccio analitico: CHSH come funzione di d_m")
print()

# Per il modello su S¹, il CHSH ottimale è:
# S = 3F(π/4) − F(3π/4) dove F(Δθ) è il correlatore
# Con solo armoniche dispari: F(Δθ) = Σ_k a_{2k+1} cos((2k+1)Δθ)

# Ordine di memoria 2 (Gaussiano): solo a₁ significativo
# F₂(Δθ) ≈ a₁ cos(Δθ) → S₂ = 3a₁cos(π/4) − a₁cos(3π/4) = 4a₁/√2 = 2√2·a₁

# Ma ρ ≥ 0 impone |a₁| ≤ 1 e Σ_k |a_{2k+1}| ≤ 1 (bound di Fourier)

# Con memoria d_m = 3 (non-Gaussiano): anche a₃ contribuisce
# F₃(Δθ) = a₁cos(Δθ) + a₃cos(3Δθ) → S₃ più complicato

a1_cl = fourier_coeffs[1][3]  # ampiezza della prima armonica

# CHSH per ogni troncamento armonico
print(f"  {'Armoniche':>14} {'a₁':>8} {'a₃':>8} {'a₅':>8} {'S_CHSH':>8} {'vs 2√2':>8}")
print(f"  {'─'*54}")

for n_harm in [1, 2, 3, 4]:
    # Ricostruisci F con n_harm armoniche dispari
    def F_trunc(theta, n_h=n_harm):
        val = 0
        for i in range(n_h):
            k = 2*i + 1
            if k < len(fourier_coeffs):
                val += fourier_coeffs[k][1] * np.cos(k*theta)
        return val
    
    # Ottimizza CHSH sugli angoli
    best_S = 0
    for x in np.linspace(0.3, 1.2, 20):
        S = abs(F_trunc(x) + F_trunc(x) + F_trunc(x+np.pi/2) - F_trunc(x + x + np.pi/2))
        # Standard: a₀=0, a₁=x, b₀=x/2, b₁=-x/2
        S2 = abs(F_trunc(np.pi/4) + F_trunc(-np.pi/4) + 
                 F_trunc(np.pi/4) - F_trunc(3*np.pi/4))
        best_S = max(best_S, S2)
    
    harms = [fourier_coeffs[2*i+1][1] if 2*i+1 < len(fourier_coeffs) else 0 for i in range(3)]
    print(f"  k=1..{2*n_harm-1:d}{' '*(10-len(f'k=1..{2*n_harm-1}'))} "
          f"{harms[0]:8.4f} {harms[1]:8.4f} {harms[2]:8.4f} {best_S:8.4f} "
          f"{'< 2√2' if best_S < 2*np.sqrt(2) else '≥ 2√2'}")

# ═══════════════════════════════════════════════════════════════════════
# 5. PROPOSTE SPERIMENTALI: MISURARE LA PROFONDITÀ DI MEMORIA
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'━' * 78}")
print("  5. PROPOSTE SPERIMENTALI")
print("━" * 78)

print("""
  Se il mezzo vibrazionale è reale, la profondità di memoria d_m 
  è una grandezza MISURABILE. Ecco come testarla:
  
  ┌──────────────────────────────────────────────────────────────────┐
  │  ESPERIMENTO 1: CORRELAZIONI TEMPORALI POST-SELEZIONATE         │
  │                                                                  │
  │  Idea: in un esperimento di Bell, misurare non solo E(a,b)     │
  │  ma anche le AUTOCORRELAZIONI TEMPORALI dei risultati:          │
  │                                                                  │
  │  G₂(τ) = ⟨A(t)A(t+τ)⟩   [autocorrelazione di un lato]        │
  │  G₃(τ₁,τ₂) = ⟨A(t)A(t+τ₁)A(t+τ₁+τ₂)⟩  [3 tempi]           │
  │                                                                  │
  │  QM standard: G₂ dipende solo dalla sorgente (Poisson/termica) │
  │  Modello mezzo: G₂ ha struttura specifica con τ_c = ℏ/gap      │
  │                                                                  │
  │  PREDIZIONE: se il mezzo esiste, G₂ ha un τ_c NON spiegabile   │
  │  solo dalla sorgente → firma della dinamica del mezzo.          │
  │                                                                  │
  │  Realizzazione: conteggio di coincidenze con timing sub-ns      │
  │  in esperimenti tipo Aspect/Weihs con alta statistica (~10⁸).   │
  ├──────────────────────────────────────────────────────────────────┤
  │  ESPERIMENTO 2: BELL CON RITARDO VARIABILE                      │
  │                                                                  │
  │  Idea: variare il ritardo temporale Δt tra la creazione della   │
  │  coppia e la misura. Se il mezzo ha tempo di rilassamento τ_c,  │
  │  la violazione di Bell dovrebbe CRESCERE con Δt fino a τ_c      │
  │  (il mezzo raggiunge l'equilibrio) e poi stabilizzarsi.         │
  │                                                                  │
  │  QM standard: S_CHSH è INDIPENDENTE da Δt (stati puri).        │
  │  Modello mezzo: S_CHSH(Δt) = S_∞ · (1 − exp(−Δt/τ_c))        │
  │                                                                  │
  │  PREDIZIONE: una dipendenza temporale della violazione di Bell  │
  │  con scala τ_c ≈ ℏ/(E₁−E₀) sarebbe una firma del mezzo.       │
  │                                                                  │
  │  Realizzazione: PDC con gating temporale variabile.             │
  ├──────────────────────────────────────────────────────────────────┤
  │  ESPERIMENTO 3: SPETTRO DI POTENZA DELLE COINCIDENZE           │
  │                                                                  │
  │  Idea: il rate di coincidenze R(t) in un esperimento di Bell    │
  │  ha uno SPETTRO DI POTENZA S(ω) = |FT[R(t)]|².                 │
  │                                                                  │
  │  QM standard: S(ω) è piatto (shot noise) o Lorentziano.        │
  │  Modello mezzo: S(ω) ha PICCHI a ω = (E_n − E_0)/ℏ,           │
  │  corrispondenti alle transizioni del mezzo vibrazionale.         │
  │                                                                  │
  │  PREDIZIONE: picchi spettrali a frequenze specifiche            │
  │  ω₁ = gap/ℏ, ω₂ = 2gap/ℏ, ...                                 │
  │                                                                  │
  │  Realizzazione: analisi FFT dei dati di timing ad alta          │
  │  risoluzione in esperimenti Bell esistenti.                      │
  └──────────────────────────────────────────────────────────────────┘
""")

# Calcolo numerico delle predizioni
gap = 0.1148  # E₁ − E₀ dal nostro Hamiltoniano
omega_1 = gap / h2_2m  # in unità del modello

print(f"  Predizioni numeriche:")
print(f"    Gap spettrale: E₁−E₀ = {gap:.4f}")
print(f"    Frequenza fondamentale: ω₁ = {omega_1:.4f} (unità del modello)")
print(f"    Tempo di rilassamento: τ_c ≈ 1/ω₁ = {1/omega_1:.1f}")

# ═══════════════════════════════════════════════════════════════════════
# 6. IL DIAGRAMMA UNIFICANTE: FOURIER-BELL-MEMORIA
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'━' * 78}")
print("  6. IL DIAGRAMMA UNIFICANTE")
print("━" * 78)

print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │              PROFONDITÀ DI MEMORIA d_m                          │
  │                    ↓                                            │
  │         ┌─────────┴─────────┐                                  │
  │         │                   │                                  │
  │      d_m = 2             d_m = ∞                               │
  │    (Markoviano)      (QM completa)                             │
  │         │                   │                                  │
  │    Solo C₂(τ)        Tutte C_n(τ...)                          │
  │         │                   │                                  │
  │    ρ(φ) ≥ 0           ψ(φ) ∈ C                                │
  │  (distrib. classica)  (ampiezza)                               │
  │         │                   │                                  │
  │  ┌──────┴──────┐    ┌──────┴──────┐                           │
  │  │ Settore     │    │ Tutti i     │                           │
  │  │ armonico    │    │ settori     │                           │
  │  │ PARI solo   │    │ armonici    │                           │
  │  │ (T4)        │    │ accessibili │                           │
  │  └──────┬──────┘    └──────┬──────┘                           │
  │         │                   │                                  │
  │    CHSH ≤ 2            CHSH = 2√2                             │
  │    Cross = 0           Cross ≠ 0                              │
  │    NS geometrico       NS tensoriale                          │
  │         │                   │                                  │
  │    MODELLO S¹          QM su L²(S¹)                           │
  │         │                   │                                  │
  │         └─────────┬─────────┘                                  │
  │                   │                                            │
  │         MISURABILE: profilo di                                 │
  │         G₂(τ), G₃(τ₁,τ₂), S(ω)                              │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘
  
  La profondità di memoria d_m è la grandezza che INTERPOLA 
  tra il modello classico (d_m = 2) e la QM (d_m = ∞).
  
  Il valore di d_m per un sistema fisico reale è MISURABILE 
  attraverso le correlazioni temporali dei risultati di Bell.
  
  Se d_m > 2 per i fotoni reali → il mezzo ha memoria non-Markoviana.
  Se d_m = 2 → il mezzo è puramente termico (Markoviano).
  Se d_m = ∞ → la QM standard è la descrizione completa.
  
  IL TEST SPERIMENTALE DISCRIMINANTE:
  
  Misurare G₃(τ₁,τ₂) in un esperimento di Bell ad alta statistica.
  Se G₃ ≠ 0 e ha la struttura prevista dal potenziale cos(4φ):
  → il mezzo esiste e ha profondità di memoria finita.
  Se G₃ = 0 a ogni scala temporale:
  → nessuna evidenza del mezzo (QM standard o d_m = ∞).
""")

# ═══════════════════════════════════════════════════════════════════════
# 7. PREDIZIONI QUANTITATIVE
# ═══════════════════════════════════════════════════════════════════════
print(f"{'━' * 78}")
print("  7. PREDIZIONI QUANTITATIVE PER FOTONI REALI")
print("━" * 78)

# Se ℏ = 1.055×10⁻³⁴ J·s e il gap è E₁−E₀ per il potenziale cos(4φ):
# τ_c = ℏ/gap ≈ ℏ/(E₁−E₀)
# Per fotoni ottici (λ = 800nm): E_fotone ≈ 1.55 eV
# Il "gap del mezzo" sarebbe una frazione dell'energia del fotone

# Stime ordine di grandezza
hbar_SI = 1.055e-34  # J·s
E_photon_eV = 1.55   # eV per fotone 800nm
E_photon_J = E_photon_eV * 1.602e-19

# Se il gap del mezzo scala come gap/E_photon ≈ gap_adim ≈ 0.11
gap_fraction = gap / abs(c2)  # rapporto gap / scala del potenziale
tau_c_estimate = hbar_SI / (gap_fraction * E_photon_J)

print(f"  Stime ordine di grandezza:")
print(f"    Gap adimensionale: {gap:.4f}")
print(f"    Rapporto gap/V: {gap_fraction:.4f}")
print(f"    Per fotoni a 800nm (E = {E_photon_eV} eV):")
print(f"    τ_c stimato ≈ {tau_c_estimate:.2e} s")
print(f"    ω₁ stimato ≈ {1/tau_c_estimate:.2e} Hz")
print(f"    → {'Nel range di risoluzione temporale attuale (sub-ps)' if tau_c_estimate > 1e-15 else 'Sotto la risoluzione attuale'}")

# Frequenze spettrali previste
print(f"\n  Picchi spettrali previsti nello spettro di coincidenze:")
for n in range(1, 5):
    omega_n = n * 1/tau_c_estimate
    period_n = 1/omega_n
    print(f"    n={n}: ω = {omega_n:.2e} Hz, T = {period_n:.2e} s")

print(f"""
  ★ CONCLUSIONE:
  
  La connessione Fourier-Bell-Memoria definisce un programma 
  sperimentale concreto:
  
  1) Misurare G₂(τ) nei dati di timing di esperimenti Bell
  2) Cercare picchi in S(ω) a frequenze specifiche
  3) Verificare se S_CHSH dipende dal ritardo Δt
  4) Misurare G₃ per stimare d_m
  
  Se d_m > 2: il mezzo vibrazionale ha una firma osservabile.
  Se d_m = 2: il modello classico è indistinguibile dal Markoviano.
  Se la dipendenza da Δt è trovata: il mezzo ha una scala temporale.
  
  Questi test sono realizzabili con la tecnologia attuale 
  (timing sub-ns, alta statistica, analisi FFT post-processing).
""")

