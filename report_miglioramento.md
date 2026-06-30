# Report di Analisi e Piano di Implementazione
## Progetto Acrobot — Optimal Control (Versione Base: V4UntilTask4)

---

## 1. Stato Attuale — Cosa Funziona

| Componente | Stato | Note |
|---|---|---|
| `dynamics.py` | ✅ Solido | RK4 + FD Jacobiani. Parametri Set 3 corretti |
| `equilibrium_finding.py` | ✅ Solido | SQP con KKT System corretto |
| `cost.py` | ✅ Solido | Costo quadratico con Q/R/Q_T ben calibrati |
| `armijo.py` | ✅ Solido | Open-loop e closed-loop update |
| `solver_newton.py` | ✅ Solido | Q-function expansion con regolarizzazione |
| `reference_trajectory.py` | ✅ Solido | Gradino e smooth quintico |
| Task 1 (main Newton) | ✅ Funzionante | Warm start corretto. Convergenza verificata |
| Task 2 (smooth ref.) | ⚠️ Funzionante | Concatenazione ref. OK ma max_iters basso |
| Task 3 (TV-LQR) | ✅ Funzionante | Riccati backward OK. Legge tracking corretta |
| Task 4 (MPC CasADi) | ⚠️ Funzionante | Terminal cost da Riccati. Ma codice duplicato |
| Task 5 (animazione) | 🐛 Bug | Import da `dynamics_acrobot` invece di `dynamics` |

---

## 2. Problemi Identificati — Per Priorità

### 🔴 CRITICI (bloccano correttezza o esecuzione)

#### P1 — Bug Import in `task5_animation.py`
```python
# SBAGLIATO (riga 4):
from dynamics_acrobot import dynamics

# CORRETTO:
from dynamics import dynamics
```
Il file `dynamics_acrobot` appartiene a V1 (parametri diversi, interfaccia legacy).
Se non corretto, l'animazione userà dinamica sbagliata o andrà in crash.

#### P2 — Riccati LQR in Task 3: forma non standard
La Task 3 usa la forma compatta:
```
P = Q + (A-BK)ᵀ P (A-BK) + Kᵀ R K
```
Questa è la **Lyapunov equation del sistema in closed-loop**, non la DARE diretta.
È matematicamente equivalente ma computazionalmente meno stabile.
La forma standard raccomandabile dal prof è (Slide 10):
```
P_t = Q + Aᵀ P_{t+1} A - Aᵀ P_{t+1} B (R + Bᵀ P_{t+1} B)⁻¹ Bᵀ P_{t+1} A
K_t = (R + Bᵀ P_{t+1} B)⁻¹ Bᵀ P_{t+1} A
```
E il modulo `ltv_LQR` di CodingSession5 implementa esattamente questo.

#### P3 — `max_iters = 3` in `task2_main` (V3 era così, V4 lo ha aumentato a 50 ma va verificato)
Verificare che Task 2 converga effettivamente. 3 iterazioni Newton non sono sufficienti
per lo swing-up che è un problema non-convesso con molte iterazioni necessarie.

---

### 🟡 IMPORTANTI (degradano la qualità del codice e la presentazione)

#### P4 — Codice Duplicato tra Task 3 e Task 4
La linearizzazione e il backward Riccati sono reimplementati inline in entrambi i task:
- `task3_main_V3.py`: backward Riccati inline (~15 righe)
- `task4.casadi.py`: backward Riccati inline (~10 righe)

**Soluzione**: estrarre in un modulo `solver_ltv_lqr.py` basato sul codice di CodingSession5 (`4_solver_ltv_LQR.py`). Esempio di interfaccia:
```python
# solver_ltv_lqr.py  (da creare — riadattare da CodingSession5)
def ltv_LQR(AA, BB, QQ, RR, SS, QQf, TT, x0, ...):
    ...
    return KK, sigma, PP, xx_out, uu_out
```

#### P5 — Naming Confuso (`task3_main_V3.py` in V4)
Il file si chiama `_V3` ma vive in V4. Rinominare in `task3_main.py`.

#### P6 — Convergenza non uniforme tra Task 1 e Task 2
Task 1 ha un criterio `abs(ΔJ) < 1e-4`. Task 2 usa solo `max_iters`.
Il professore usa `descent[kk] <= term_cond = 1e-4` (norma del gradiente).
Uniformare il criterio di stop alla norma del descent: `||Δu|| < ε`.

#### P7 — Mancanza del Plot di Armijo in Task 2
Task 1 mostra il plot Armijo per le prime 3 iterazioni (`plot=(k < 3)`).
Task 2 non lo mostra. Utile per diagnosticare problemi di convergenza.

#### P8 — Terminal Cost in Task 4 non ottimale al bordo
```python
P_term = P_list[idx_end-1] if idx_end < steps else Q_mpc
```
Quando `idx_end == steps` usa `Q_mpc` come fallback (non il Riccati).
Dovrebbe usare `P_list[-1]` (l'ultimo elemento calcolato).

---

### 🟢 MIGLIORAMENTI OPZIONALI (aumentano la qualità del progetto)

#### M1 — Aggiungere DARE per Q_T (come nel prof Session 4)
Il professore calcola `QQT = ctrl.dare(AA_ref, BB_ref, QQt, RRt)[0]`.
Questo garantisce che Q_T approssimi il costo infinito-orizzonte all'equilibrio,
rendendo la soluzione a orizzonte finito meno dipendente dalla scelta manuale di Q_T.

#### M2 — Plot iterativo durante la convergenza (stile professore)
Il prof (Session 4, righe 163-303) aggiorna il plot ad ogni iterazione con `plt.pause(1e-4)`.
Permette di vedere la traiettoria evolversi in tempo reale — molto utile per la presentazione.

#### M3 — Multiple Initial Conditions Test per Task 3
V1 testava due perturbazioni diverse (`[-0.2,0,0,0]` e `[0, 0.3, 0, 0]`) per mostrare
la robustezza del TV-LQR. V4 Task 3 usa solo una. Aggiungere il secondo test.

#### M4 — Normalizzazione angoli in `forward_kinematics`
L'animazione non gestisce `θ₁ > 2π` o `θ₁ < -2π`. Aggiungere:
```python
th1 = th1 % (2 * np.pi)  # Wrap-around per angoli
```

#### M5 — Salvataggio della traiettoria Task 4 (MPC)
Task 4 non salva i risultati. Utile per analisi post-hoc o per un eventuale Task 5 MPC.

---

## 3. Piano di Implementazione

### Fase 1 — Bug Fix Critici (1-2 ore)

- [ ] **P1**: Correggere import in `task5_animation.py` → `from dynamics import dynamics`
- [ ] **P3**: Verificare e impostare `max_iters ≥ 50` in task2_main
- [ ] **P8**: Correggere fallback terminal cost in Task 4: `P_list[-1]` invece di `Q_mpc`

### Fase 2 — Refactoring e Uniformità (2-3 ore)

- [ ] **P4**: Creare `solver_ltv_lqr.py` da CodingSession5 (`4_solver_ltv_LQR.py`)
  - Interfaccia: `ltv_LQR(AA, BB, QQ, RR, SS, QQf, TT, x0)` → `KK, sigma, PP`
  - Sostituire il codice inline in Task 3 e Task 4

- [ ] **P2**: Riscrivere backward Riccati in Task 3 con la forma standard:
  ```python
  # Forma standard (da CodingSession5)
  S   = R + B.T @ P @ B
  K_t = np.linalg.solve(S, B.T @ P @ A)
  P   = Q + A.T @ P @ A - (A.T @ P @ B) @ K_t
  ```

- [ ] **P5**: Rinominare `task3_main_V3.py` → `task3_main.py`

- [ ] **P6**: Uniformare criterio di convergenza in Task 1 e Task 2:
  ```python
  # Usare norma descent come nel professore
  if np.linalg.norm(kk_vec) < term_cond:
      print("Convergenza raggiunta (||Δu|| < ε).")
      break
  ```

- [ ] **P7**: Aggiungere `plot=(k < 3)` ad Armijo call in Task 2

### Fase 3 — Miglioramenti Teoria (2-3 ore)

- [ ] **M1**: Calcolare Q_T con DARE nel main Task 1 e Task 2:
  ```python
  import control as ctrl
  # Linearizzazione all'equilibrio obiettivo
  _, A_eq, B_eq = dyn.dynamics(x_goal, u_goal)
  QQT = ctrl.dare(A_eq, B_eq, cst.QQt, cst.RRt)[0]
  ```

- [ ] **M2**: Aggiungere plot iterativo (stile professore) in Task 1 e Task 2

- [ ] **M3**: Aggiungere secondo test di perturbazione in Task 3

### Fase 4 — Rifinitura e Presentazione (1-2 ore)

- [ ] **M4**: Aggiungere wrap angoli nell'animazione
- [ ] **M5**: Aggiungere salvataggio risultati Task 4
- [ ] Aggiungere commenti teorici ai file ancora privi (task2, task3, task4)
- [ ] Verificare che la pipeline completa (`equilibrium_finding` → `task1` → `task2` → `task3` → `task4` → `task5`) giri senza errori
- [ ] Aggiungere un `README.md` con istruzioni di esecuzione e dipendenze

---

## 4. Checklist di Verifica Tecnica

Prima della presentazione, verificare:

- [ ] `equilibrium_finding.py` converge con `constraint_norm < 1e-6` per entrambi gli equilibri
- [ ] Task 1: il costo J scende monotonicamente nel plot di convergenza
- [ ] Task 1: la traiettoria ottima raggiunge `x_goal ≈ [π, 0, 0, 0]` a t=T
- [ ] Task 2: il riferimento smooth è continuo e differenziabile (no discontinuità visibili)
- [ ] Task 2: la traiettoria ottima segue il smooth reference (non il gradino)
- [ ] Task 3: con perturbazione iniziale di 0.2 rad, il tracking error va a zero
- [ ] Task 4: il grafico dell'errore di tracking (MPC) è inferiore a Task 3 (LQR)
- [ ] Task 5: l'animazione parte dal basso (θ₁≈0) e termina in alto (θ₁≈π)

---

## 5. Dipendenze Python (da installare se mancanti)

```bash
pip install numpy scipy matplotlib sympy casadi control
```

> [!IMPORTANT]
> CasADi con IPOPT è necessario per Task 4. Su Windows, installare con:
> `pip install casadi` (include IPOPT precompilato)

> [!NOTE]
> Il pacchetto `control` è usato per `ctrl.dare()` (Fase 3, M1).
> Installare con `pip install control`

---

## 6. Struttura File Raccomandata (Progetto Finale)

```
Project/
├── dynamics.py              ← Dinamica RK4 + Jacobiani (INVARIATA)
├── cost.py                  ← Costo quadratico Q/R/Q_T (verificare DARE)
├── armijo.py                ← Line search Armijo (INVARIATA)
├── solver_newton.py         ← Backward pass iDDP (INVARIATA)
├── solver_ltv_lqr.py        ← [NUOVO] Backward Riccati / LTV-LQR standard
├── reference_trajectory.py  ← Gradino + Smooth quintico (INVARIATA)
├── equilibrium_finding.py   ← SQP per equilibri (INVARIATA)
├── equilibrium_data.npy     ← Dati equilibri precalcolati
├── task0_equilibrium.py     ← [OPZIONALE] Script autonomo per task 0
├── task1_main.py            ← Task 1: Newton/iDDP swing-up (step ref.)
├── task2_main.py            ← Task 2: Newton/iDDP swing-up (smooth ref.)
├── task3_main.py            ← Task 3: TV-LQR tracking [rinominato]
├── task4_main.py            ← Task 4: MPC CasADi tracking [rinominato]
├── task5_animation.py       ← Animazione [fix import]
└── README.md                ← [NUOVO] Istruzioni
```
