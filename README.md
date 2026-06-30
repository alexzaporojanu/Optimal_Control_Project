# Acrobot Optimal Control — V5 Final
## Progetto: Corso di Optimal Control & Reinforcement Learning
### UniBO 2025/26 — Parameter Set 3

---

## Struttura del Progetto

```
V5_Final/
│
├── 📦 MODULI CORE (condivisi tra tutti i Task)
│   ├── dynamics.py            ← Dinamica Acrobot (Eulero-Lagrange + RK4)
│   ├── cost.py                ← Funzioni di costo (stage, terminale, DARE)
│   ├── armijo.py              ← Armijo line search (Open-Loop e Closed-Loop)
│   ├── solver_newton.py       ← Backward pass iDDP / Newton
│   ├── solver_ltv_lqr.py      ← [NUOVO V5] Backward Riccati TV-LQR standard
│   ├── reference_trajectory.py ← Generatore riferimenti (gradino, smooth, extended)
│   └── equilibrium_finding.py  ← SQP per ricerca equilibri (KKT system)
│
├── 🎯 TASK PRINCIPALI
│   ├── task1_main.py          ← Task 1: iDDP con riferimento GRADINO
│   ├── task2_main.py          ← Task 2: iDDP con riferimento SMOOTH (3 fasi)
│   ├── task3_main.py          ← Task 3: TV-LQR Tracking
│   ├── task4_main.py          ← Task 4: LTV-MPC Tracking (CasADi + IPOPT)
│   └── task5_animation.py     ← Task 5: Animazione (cinematica diretta)
│
└── 📊 FILE GENERATI (al runtime)
    ├── equilibrium_data.npy        ← Equilibri pre-calcolati
    ├── optimal_trajectory_task1.npy ← Traiettoria ottima Task 1
    ├── optimal_trajectory_task2.npy ← Traiettoria ottima Task 2
    ├── lqr_data_task3.npy          ← Guadagni K_t e Riccati P_t (Task 3)
    ├── mpc_results_task4.npy       ← Risultati MPC (Task 4)
    └── acrobot_animation.gif       ← Animazione (opzionale)
```

---

## Ordine di Esecuzione

I task vanno eseguiti in ordine. Ogni task dipende dall'output del precedente.

```bash
# 1. Calcola gli equilibri (x_eq1, x_eq2)
python equilibrium_finding.py

# 2. Task 1: traiettoria ottima con riferimento a gradino
python task1_main.py

# 3. Task 2: traiettoria ottima con riferimento smooth (PREFERITA per Task 3/4)
python task2_main.py

# 4. Task 3: TV-LQR tracking (usa Task 2 come riferimento)
python task3_main.py

# 5. Task 4: MPC tracking con CasADi (usa Task 2 + Riccati di Task 3)
python task4_main.py

# 6. Task 5: Animazione
python task5_animation.py
```

---

## Dipendenze Python

Installa le dipendenze con:
```bash
pip install numpy scipy matplotlib sympy casadi control pillow
```

| Pacchetto | Versione consigliata | Uso |
|---|---|---|
| `numpy` | ≥ 1.24 | Calcolo numerico |
| `scipy` | ≥ 1.11 | `solve_ivp`, `block_diag` |
| `matplotlib` | ≥ 3.7 | Plot e animazione |
| `sympy` | ≥ 1.12 | Compilazione matrici simboliche |
| `casadi` | ≥ 3.6 | Solver MPC (Task 4) |
| `control` | ≥ 0.9 | DARE per costo terminale Q_T |
| `pillow` | ≥ 10.0 | Salvataggio GIF (opzionale, Task 5) |

> **Nota**: `casadi` su Windows include già IPOPT precompilato. Non è necessario installarlo separatamente.

---

## Parametri Fisici — Set 3

| Parametro | Valore | Unità |
|---|---|---|
| m₁, m₂ | 1.5 | kg |
| l₁, l₂ | 2.0 | m |
| lc₁, lc₂ | 1.0 | m |
| I₁, I₂ | 2.0 | kg·m² |
| g | 9.81 | m/s² |
| f₁, f₂ | 1.0 | N·m·s/rad |
| dt | 0.01 | s |

---

## Riferimenti Teorici (Slide)

| Slide | Contenuto | Usato in |
|---|---|---|
| Slide 03 | KKT Conditions | `equilibrium_finding.py` |
| Slide 04 | Discrete LQR / Riccati Equation | `solver_ltv_lqr.py` |
| Slide 05 | Dynamical Systems / Euler-Lagrange | `dynamics.py` |
| Slide 06 | Optimal Control Shooting | `reference_trajectory.py`, Task 1 |
| Slide 07 | Gradient Method / Armijo | `armijo.py`, Task 1, Task 2 |
| Slide 08 | Second-Order Methods / iDDP | `solver_newton.py`, Task 1, Task 2 |
| Slide 10 | Optimal Control Tracking / TV-LQR | `task3_main.py` |
| Slide 11 | Model Predictive Control | `task4_main.py` |

---

## Miglioramenti rispetto a V4

| # | Problema | Fix V5 |
|---|---|---|
| P1 | Bug import `task5_animation.py` | → `from dynamics import dynamics` |
| P2 | Riccati in Task 3 in forma non-standard | → Forma FORM 1 stabile in `solver_ltv_lqr.py` |
| P3 | `max_iters = 3` in Task 2 | → `max_iters = 50` uniforme |
| P4 | Codice Riccati duplicato in Task 3/4 | → Modulo condiviso `solver_ltv_lqr.py` |
| P5 | Naming `task3_main_V3.py` in V4 | → Rinominato `task3_main.py` |
| P6 | Convergenza non uniforme Task 1/2 | → `descent[kk] ≤ term_cond` ovunque |
| P7 | Mancanza plot Armijo in Task 2 | → Aggiunto `plot=(kk < 3)` |
| P8 | Terminal cost Task 4 non ottimale | → P_list dalla Riccati di Task 3 |
| M1 | Q_T ad-hoc (20000·I) | → DARE all'equilibrio obiettivo |
| M2 | Mancanza plot iterativo Task 2 | → Aggiunto plot iterativo stile professore |
| M3 | Una sola perturbazione in Task 3/4 | → 3 perturbazioni diverse |
| M4 | No wrap angoli in animazione | → `theta % (2π)` in forward kinematics |
| M5 | Mancanza salvataggio Task 4 | → Salvataggio in `mpc_results_task4.npy` |
