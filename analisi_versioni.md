# Analisi Comparativa — Progetto Acrobot Optimal Control

> Metriche di valutazione: **Correttezza Logica & Teoria** · **Avanzamento dei Lavori** · **Chiarezza Codice** · **Aderenza agli Esempi del Professore**

---

## Versione 1 — `Step1_vers1/Project/`

### File presenti
`dynamics_acrobot.py`, `task1_corretta`, `task1_Newton`, `task1_newton_s_armijo`, `task2_trajectory_generation.py`, `task3_lqr_tracking.py`, `task4_mpc_tracking1.py`, `armijo.py`, `cost.py`, `equilibrium_finding.py`

### ✅ Pro
- **Avanzamento più ampio**: copre tutti e 4 i task (Task 1-4) incluso MPC con CVXPY
- **Task 3 TV-LQR**: implementazione della legge di tracking `u = u_ref + K(x - x_ref)` formalmente corretta
- **Task 4 MPC con CVXPY**: uso corretto della formulazione lineare in deviazione con offset `x_{k+1} - x_ref = A(x - x_ref) + B(u - u_ref)`, incluso multi-perturbazione per test di robustezza
- **dynamics_acrobot.py**: parametri raccolti in dizionario `p{}` — buona leggibilità
- **Struttura self-contained**: ogni Task è in un file separato

### ❌ Contro
- **Task 1**: varianti multiple confuse (`task1_corretta`, `task1_Newton`, `task1_newton_s_armijo`) senza un file definitivo; indica incertezza su quale implementazione fosse corretta
- **Warm start**: il "kick" sinusoidale è applicato tra t=2s e t=4s, ma il gradino avviene a T/2=10s — il kick è lontano dalla transizione critica, riducendo l'efficacia
- **Backward pass** in `task1_corretta`: calcola `Q_ux = A.T @ V_xx @ B` ma poi usa `Q_ux.T` per `K` — c'è un'inconsistenza rispetto alla formulazione standard dove `Q_{xu} = Q_{ux}^T`
- **Task 4 terminal cost**: usa `P_terminal = Q * 10.0` — approssimazione grossolana invece di usare la soluzione della DARE/Riccati backward
- **Aderenza al professore**: l'armijo in task1 è reimplementato inline nel main loop invece di usare il modulo `armijo.py` già disponibile; discrepanza con il pattern del professore
- **Naming non uniforme**: `dynamics_acrobot.py` vs `dynamics.py` nelle versioni successive
- **Commenti**: quasi assenti nelle parti teoricamente critiche (backward pass, Riccati)

### 📊 Punteggi
| Criterio | Voto |
|---|---|
| Correttezza Logica & Teoria | ⭐⭐⭐ (3/5) |
| Avanzamento dei Lavori | ⭐⭐⭐⭐ (4/5) |
| Chiarezza Codice | ⭐⭐ (2/5) |
| Aderenza agli Esempi del Prof | ⭐⭐ (2/5) |

---

## Versione 2 — `Step1_V2(Golden_alex)/Project_golden/`

### File presenti
`dynamics.py`, `armijo.py`, `cost.py`, `equilibrium_finding.py`, `reference_trajectory.py`, `solver_newton.py`, `task1_main_newton.py`

### ✅ Pro
- **Architettura modulare eccellente**: ogni componente è in un file separato e riutilizzabile — `dynamics.py`, `armijo.py`, `cost.py`, `solver_newton.py`, `reference_trajectory.py` — perfettamente in linea con il pattern del professore
- **`solver_newton.py`**: implementazione pulita del backward pass iDDP-Newton con Q-function expansion completa (`Qx, Qu, Qxx, Quu, Qux`)
- **Regularizzazione Levenberg-Marquardt**: `Quu_reg = Quu + 1e-2 * I` — teoricamente giustificata e necessaria per robustezza
- **`equilibrium_finding.py`**: algoritmo SQP completo con KKT system `[B dh^T; dh 0][dz; λ] = [-dl; -hh]` — formalmente corretto
- **`armijo.py`**: supporta sia Open-Loop (Gradient) che Closed-Loop (Newton), gestendo `K_fb` come parametro opzionale — riutilizzabile
- **`reference_trajectory.py`**: implementa sia gradino che smooth con polinomio quintico
- **Warm start**: sinusoide correttamente posizionata vicino alla transizione (t=4-6s per gradino a T/2=5s)

### ❌ Contro
- **Solo Task 1**: è la versione più avanzata ma copre solo il primo task — mancano Task 2, 3, 4
- **`dynamics.py`**: usa differenze finite centrali con `eps=1e-5` per i Jacobiani — corretto ma non usa la sessione 4 del prof (differenze finite one-sided)
- **Convergenza check**: `abs(JJ[k] - JJ[k-1]) < 1e-4` è un criterio di arresto basato sulla variazione di costo, non sulla norma del gradiente come nel prof (`descent[kk] <= term_cond`)
- **Print-based debugging** ristretto — il prof usa plot iterativi `plt.pause()` per monitorare la convergenza

### 📊 Punteggi
| Criterio | Voto |
|---|---|
| Correttezza Logica & Teoria | ⭐⭐⭐⭐ (4/5) |
| Avanzamento dei Lavori | ⭐⭐ (2/5) |
| Chiarezza Codice | ⭐⭐⭐⭐ (4/5) |
| Aderenza agli Esempi del Prof | ⭐⭐⭐⭐ (4/5) |

---

## Versione 3 — `V3_Project_golden/Project_golden_new/`

### File presenti
`dynamics.py`, `armijo.py`, `cost.py`, `equilibrium_finding.py`, `reference_trajectory.py`, `solver_newton.py`, `task1_main_newton.py`, `task2_main`

### ✅ Pro
- **Estende V2 con Task 2**: aggiunge la generazione di traiettoria con riferimento smooth
- **Struttura modulare mantenuta**: eredita tutta la bontà di V2
- **Task 2 Extended**: costruisce un riferimento a 3 fasi (Pre-wait → Smooth Move → Post-hold) con concatenazione corretta
- **Warm start** di Task 2 centrato sulla fase di movimento: `uu[0, N_pre:N_pre+N_move, 0] = 5 * sin(3t)` — teoricamente sensato
- **Salvataggio dati**: in V2 mancava il salvataggio per Task successive; qui è presente

### ❌ Contro
- **max_iters = 3** in task2_main: valore irrisorio, probabilmente rimasto da un test; 3 iterazioni Newton non sono sufficienti per convergenza
- **Concatenazione reference**: `xx_ref = np.hstack([xx_pre, xx_move[:, :-1], xx_post])` poi sovrascritto da `xx_ref = np.hstack([xx_pre, xx_move, xx_post[:, 1:]])` — codice morto; indicatore di sviluppo non rifinito
- **Commenti ridondanti e contraddittori**: blocchi di commento che si contraddicono ("Tagliamo 1 da move per continuità o aggiustiamo" poi corretto sopra)
- **Task 3 e 4 mancanti**

### 📊 Punteggi
| Criterio | Voto |
|---|---|
| Correttezza Logica & Teoria | ⭐⭐⭐⭐ (4/5) |
| Avanzamento dei Lavori | ⭐⭐⭐ (3/5) |
| Chiarezza Codice | ⭐⭐⭐ (3/5) |
| Aderenza agli Esempi del Prof | ⭐⭐⭐⭐ (4/5) |

---

## Versione Step3_v01 — `Step3_v01/`

### File presenti
`dynamics.py`, `task3_main_V3.py` (solo)

### ✅ Pro
- **Task 3 TV-LQR**: implementazione del backward pass di Riccati con equazione corretta in forma compatta `P = Q + (A-BK)^T P (A-BK) + K^T R K`
- **Fix robusto dimensioni**: gestione automatica dei casi (N,4) vs (4,N) — buona pratica
- **Legge di controllo corretta**: `u = u_ref - K * delta_x`

### ❌ Contro
- **Versione monca**: manca di tutti gli altri moduli (`armijo.py`, `cost.py`, ecc.) — non è un progetto self-contained; è solo un frammento
- **Import da `dynamics`**: dipende dal file V4 nella stessa cartella, non è coerente con la struttura degli altri moduli
- **Nessun Task 1, 2, 4**: impossibile valutare il progetto nella sua completezza

### 📊 Punteggi
| Criterio | Voto |
|---|---|
| Correttezza Logica & Teoria | ⭐⭐⭐⭐ (4/5) |
| Avanzamento dei Lavori | ⭐ (1/5) |
| Chiarezza Codice | ⭐⭐⭐ (3/5) |
| Aderenza agli Esempi del Prof | ⭐⭐ (2/5) |

---

## Versione 4 — `V4UntilTask4/` ⭐ **LA PIÙ PROMETTENTE**

### File presenti
`dynamics.py`, `armijo.py`, `cost.py`, `equilibrium_finding.py`, `reference_trajectory.py`, `solver_newton.py`, `task1_main_newton.py`, `task2_main`, `task3_main_V3.py`, `task4.casadi.py`, `task5_animation.py`

### ✅ Pro
- **Copertura più ampia**: copre tutti e 5 i task (Task 1–5), inclusa animazione
- **Architettura modulare eccellente**: eredita i moduli puliti di V2/V3 con miglioramenti aggiuntivi
- **Task 3 TV-LQR**: identico a Step3_v01, formalmente corretto
- **Task 4 MPC con CasADi**: uso professionale di `ca.Opti()` con IPOPT — formulazione in error dynamics `δx_{k+1} = A δx_k + B δu_k`, vincoli sull'input assoluto `u_ref + δu ∈ [-U_MAX, U_MAX]`
- **Terminal Cost Riccati in Task 4**: `P_list` calcolato backward con la DARE per ogni istante — teoricamente corretta e superiore alla V1
- **`dynamics.py`**: check esplicito sulla dimensionalità con messaggio d'errore informativo — codice difensivo
- **`_step_only`**: helper privato per evitare ricorsione nei Jacobiani — scelta progettuale corretta
- **Task 5 Animazione**: cinematica diretta `(x0,y0) → (x1,y1) → (x2,y2)` corretta con convenzione angolare esplicitata
- **Parametri fisici Set 3**: correttamente impostati e condivisi via `dynamics.py`

### ❌ Contro
- **Riccati backward in Task 3**: calcola `K = S^{-1} B^T P A` che fornisce la legge `u = -K x`; questa forma non include il termine di *feedforward* corretto. La forma standard dell'aggiornamento di Riccati per il tracking dovrebbe includere anche il vettore lineare `p` (affine LQR). **Tuttavia** in questo progetto il tracking è fatto in deviazione (Δx), quindi è accettabile come approssimazione
- **Task 4**: non usa il `solver_newton.py` come modulo — implementa la linearizzazione inline ogni volta (codice duplicato)
- **Backward pass Riccati in Task 3 e Task 4**: le due implementazioni del backward Riccati sono leggermente diverse (`P = Q + K^T R K + (A-BK)^T P (A-BK)` vs `P = Q + A^T P A - A^T P B (R + B^T P B)^{-1} B^T P A`) — matematicamente equivalenti ma non unificate in un modulo condiviso tipo il `4_solver_ltv_LQR.py` del prof
- **Commenti**: quasi assenti nelle sezioni matematicamente dense (backward pass, Q-function expansion) — critico per la presentazione
- **Denominazione task3_main_V3.py**: il suffisso "V3" in V4 è confondente
- **`task5_animation.py`**: importa da `dynamics_acrobot` (V1) invece di `dynamics` (V4) — **bug potenziale**
- **Criteri di convergenza** non uniformi tra Task 1 e Task 2: diversi `max_iters`, uno ha convergenza basata su ΔJ l'altro solo sul limite di iterazioni

### 📊 Punteggi
| Criterio | Voto |
|---|---|
| Correttezza Logica & Teoria | ⭐⭐⭐⭐ (4/5) |
| Avanzamento dei Lavori | ⭐⭐⭐⭐⭐ (5/5) |
| Chiarezza Codice | ⭐⭐⭐ (3/5) |
| Aderenza agli Esempi del Prof | ⭐⭐⭐⭐ (4/5) |

---

## 🏆 Riepilogo Comparativo

| Versione | Teoria | Avanzamento | Chiarezza | Aderenza Prof | **Totale** |
|---|---|---|---|---|---|
| V1 `Step1_vers1` | 3/5 | 4/5 | 2/5 | 2/5 | **11/20** |
| V2 `Step1_V2 (Golden)` | 4/5 | 2/5 | 4/5 | 4/5 | **14/20** |
| V3 `V3_Project_golden` | 4/5 | 3/5 | 3/5 | 4/5 | **14/20** |
| Step3_v01 | 4/5 | 1/5 | 3/5 | 2/5 | **10/20** |
| **V4 `V4UntilTask4`** | **4/5** | **5/5** | **3/5** | **4/5** | **16/20** ✅ |

## Versione Consigliata: **V4UntilTask4**

**Motivazione**: È l'unica versione che copre l'intero progetto (Task 1→5) con una struttura modulare professionale. La correttezza teorica è solida su tutti i task critici. I margini di miglioramento sono chiari e correggibili senza riscrivere da zero. Con commenti approfonditi e uniformità del codice, diventa la base ideale per la presentazione.

