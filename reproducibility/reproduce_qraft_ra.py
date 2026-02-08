#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QRAFT-RA — Script di Riproduzione Deterministico (V13 + V14)

Questo script riproduce:
- V13 (General-MD): Valore CHSH e verifica operativa di no-signaling (SIG)
- V14 (Canale di riferimento industriale): Correlatore e BER (Bit Error Rate)

Dipendenze:
  - numpy
  - scipy (per la funzione ndtr)

Utilizzo:
  python reproduce_qraft_ra.py
  python reproduce_qraft_ra.py --M 8192
"""

import argparse
import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import numpy as np
from scipy.special import ndtr  # Funzione di ripartizione normale standard Φ(x)

# =============================================================================
# CONFIGURAZIONE (Valori Certificati dal Paper V13.5.4 / V14.9)
# =============================================================================

# Configurazione certificata V13 (General-MD)
V13_BETA = 2.8
V13_SIGMA_R = 0.005

# Angoli certificati V13 (radianti)
# Questi producono il massimo globale CHSH ~ 3.5763
V13_ANGLES: Dict[str, float] = dict(
    a0=3.975791109464942,
    a1=6.082402535074318,
    b0=4.992724225020348,
    b1=2.942427346113350,
)

# Valori target V13
V13_TARGET_CHSH = 3.576310390010
V13_SIG_THRESHOLD = 1e-14

# Angoli di riferimento V14 (Canale Industriale)
# Nota: Differiscono leggermente dagli ottimi V13 per garantire stabilità industriale
V14_ASTAR = 4.3468030917
V14_BSTAR = 3.2249097291

# Valori target V14
V14_TARGET_CORR = 0.846064877280
V14_TARGET_BER  = 0.076967561360


# =============================================================================
# MOTORE DI CALCOLO (ENGINE)
# =============================================================================

@dataclass(frozen=True)
class QRAFTRunConfig:
    beta: float
    sigma_r: float
    M: int


class QRAFTEngine:
    """
    Motore deterministico QRAFT-RA:
      - Griglia latente: phi in [0, 2π)
      - Azione contestuale: S_ctx(phi; a,b) = 1 - cos(phi-a) cos(phi-b)
      - Distribuzione di Gibbs: p(phi|a,b) ∝ exp(-beta * S_ctx)
      - Mappa di misura: A_bar(phi) = 2*Φ(cos(phi-theta)/sigma_r) - 1
      - Correlatore: E(a,b) = ∫ A_bar(phi) B_bar(phi) p(phi|a,b) dphi
    """

    def __init__(self, beta: float, sigma_r: float, M: int = 4096):
        if M < 16:
            raise ValueError("M deve essere almeno 16.")
        if sigma_r <= 0:
            raise ValueError("sigma_r deve essere > 0.")

        self.beta = float(beta)
        self.sigma_r = float(sigma_r)
        self.M = int(M)

        # Configurazione griglia latente [0, 2π)
        self.phi = np.linspace(0.0, 2.0 * np.pi, self.M, endpoint=False)
        self.dphi = (2.0 * np.pi) / self.M

    def _contextual_prob(self, a: float, b: float) -> np.ndarray:
        # Eq (5): Paesaggio dell'azione contestuale
        S_ctx = 1.0 - np.cos(self.phi - a) * np.cos(self.phi - b)

        # beta=0 -> distribuzione uniforme
        if self.beta == 0.0:
            return np.full(self.M, 1.0 / self.M, dtype=np.float64)

        # Stabilizzazione numerica: shift di S per min(S) (non cambia la distribuzione normalizzata)
        S_shifted = S_ctx - np.min(S_ctx)

        w = np.exp(-self.beta * S_shifted)
        Z = np.sum(w) * self.dphi

        if not np.isfinite(Z) or Z <= 0:
            raise FloatingPointError("Normalizzazione fallita: Z non finito o non positivo.")

        # Qui p è una discretizzazione simil-densità tale che ∑ p_i dphi = 1
        return w / Z

    def _measurement_map(self, theta: float) -> np.ndarray:
        # Eq (7): Mappa analitica del rumore tramite CDF Gaussiana
        arg = np.cos(self.phi - theta) / self.sigma_r
        # 2*Φ(x) - 1 mappa in [-1, 1]
        return 2.0 * ndtr(arg) - 1.0

    def correlator(self, a: float, b: float) -> float:
        # Eq (8): Quadratura deterministica
        p = self._contextual_prob(a, b)
        A_bar = self._measurement_map(a)
        B_bar = self._measurement_map(b)

        # Integrale via sommatoria
        E = np.sum(A_bar * B_bar * p) * self.dphi
        return float(E)

    def marginals(self, a: float, b: float) -> Tuple[float, float]:
        # Calcola le medie marginali <A> e <B> nel contesto (a,b) per i check di signaling
        p = self._contextual_prob(a, b)
        A_bar = self._measurement_map(a)
        B_bar = self._measurement_map(b)

        muA = float(np.sum(A_bar * p) * self.dphi)
        muB = float(np.sum(B_bar * p) * self.dphi)
        return muA, muB


# =============================================================================
# METRICHE & LOGICA DI VERIFICA
# =============================================================================

def compute_chsh(engine: QRAFTEngine, a0: float, a1: float, b0: float, b1: float) -> Dict[str, float]:
    E00 = engine.correlator(a0, b0)
    E01 = engine.correlator(a0, b1)
    E10 = engine.correlator(a1, b0)
    E11 = engine.correlator(a1, b1)
    S = E00 + E01 + E10 - E11
    return {"E00": E00, "E01": E01, "E10": E10, "E11": E11, "CHSH": float(S)}


def compute_sig(engine: QRAFTEngine, a0: float, a1: float, b0: float, b1: float) -> Dict[str, float]:
    # SIG_A: varia b con a fisso
    muA_00, _ = engine.marginals(a0, b0)
    muA_01, _ = engine.marginals(a0, b1)
    sigA0 = abs(muA_00 - muA_01)

    muA_10, _ = engine.marginals(a1, b0)
    muA_11, _ = engine.marginals(a1, b1)
    sigA1 = abs(muA_10 - muA_11)

    # SIG_B: varia a con b fisso
    _, muB_00 = engine.marginals(a0, b0)
    _, muB_10 = engine.marginals(a1, b0)
    sigB0 = abs(muB_00 - muB_10)

    _, muB_01 = engine.marginals(a0, b1)
    _, muB_11 = engine.marginals(a1, b1)
    sigB1 = abs(muB_01 - muB_11)

    SIG = max(sigA0, sigA1, sigB0, sigB1)
    return {
        "sigA0": float(sigA0),
        "sigA1": float(sigA1),
        "sigB0": float(sigB0),
        "sigB1": float(sigB1),
        "SIG": float(SIG),
    }


# =============================================================================
# ESECUZIONE PRINCIPALE (MAIN)
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Script di Riproduzione QRAFT-RA (V13 + V14)")
    parser.add_argument("--M", type=int, default=4096, help="Risoluzione griglia latente (default: 4096)")
    parser.add_argument("--json", action="store_true", help="Stampa payload JSON alla fine")
    args = parser.parse_args()

    # ---------- V13: General-MD ----------
    cfg13 = QRAFTRunConfig(beta=V13_BETA, sigma_r=V13_SIGMA_R, M=args.M)
    eng13 = QRAFTEngine(beta=cfg13.beta, sigma_r=cfg13.sigma_r, M=cfg13.M)

    a0, a1 = V13_ANGLES["a0"], V13_ANGLES["a1"]
    b0, b1 = V13_ANGLES["b0"], V13_ANGLES["b1"]

    chsh13 = compute_chsh(eng13, a0, a1, b0, b1)
    sig13 = compute_sig(eng13, a0, a1, b0, b1)

    # ---------- V14: Riferimento Industriale ----------
    cfg14 = QRAFTRunConfig(beta=V13_BETA, sigma_r=V13_SIGMA_R, M=args.M)
    eng14 = QRAFTEngine(beta=cfg14.beta, sigma_r=cfg14.sigma_r, M=cfg14.M)

    corr14 = eng14.correlator(V14_ASTAR, V14_BSTAR)
    ber14 = (1.0 - corr14) / 2.0

    # ---------- Report Output ----------
    print("============================================================")
    print("QRAFT-RA — RIPRODUZIONE DETERMINISTICA (V13 + V14)")
    print("============================================================")
    print(f"[AMBIENTE] numpy={np.__version__} | M={args.M} | dtype=float64")
    print()

    print("[V13 — General-MD]")
    print(f"beta={cfg13.beta} | sigma_r={cfg13.sigma_r}")
    print(f"angoli: a0={a0:.15f} a1={a1:.15f} b0={b0:.15f} b1={b1:.15f}")
    print(f"CHSH:          {chsh13['CHSH']:.12f}")
    print(f"Target CHSH:   {V13_TARGET_CHSH:.12f}")
    print(f"Delta:         {chsh13['CHSH'] - V13_TARGET_CHSH:+.3e}")
    print(f"SIG (max):     {sig13['SIG']:.3e}")
    if sig13["SIG"] < V13_SIG_THRESHOLD:
        print(">> V13 PASS: Vincolo di No-signaling soddisfatto (Precisione).")
    else:
        print(">> V13 FAIL: Signaling rilevato.")
    print()

    print("[V14 — Canale di riferimento industriale]")
    print(f"a*={V14_ASTAR:.10f} b*={V14_BSTAR:.10f}")
    print(f"Correlatore E: {corr14:.12f}")
    print(f"Target E:      {V14_TARGET_CORR:.12f}")
    print(f"Delta E:       {corr14 - V14_TARGET_CORR:+.3e}")
    print(f"BER=(1-E)/2:   {ber14:.12f}")
    print(f"Target BER:    {V14_TARGET_BER:.12f}")
    print(f"Delta BER:     {ber14 - V14_TARGET_BER:+.3e}")
    if abs(ber14 - V14_TARGET_BER) < 1e-10:
        print(">> V14 PASS: Baseline industriale confermata.")
    else:
        print(">> V14 FAIL: Deviazione troppo alta.")
    print()

    # ---------- Hash di Riproducibilità ----------
    run_fields = {
        "cfg13": asdict(cfg13),
        "V13_ANGLES": V13_ANGLES,
        "V13_CHSH": chsh13,
        "V13_SIG": sig13,
        "cfg14": asdict(cfg14),
        "V14": {"a_star": V14_ASTAR, "b_star": V14_BSTAR, "corr": corr14, "ber": ber14},
    }

    # Dump JSON deterministico per hashing
    run_json = json.dumps(run_fields, sort_keys=True, separators=(",", ":"))
    run_hash = hashlib.sha256(run_json.encode("utf-8")).hexdigest()

    print("[Hash di Riproducibilità]")
    print(f"SHA256(run_fields JSON) = {run_hash}")

    if args.json:
        print("\n[JSON payload]")
        print(run_json)


if __name__ == "__main__":
    main()
