# Log dei Cambiamenti — V6 Final
Questo documento traccia le modifiche apportate alla versione **V6_Final** a partire dalla versione *V5_Final* per soddisfare alla lettera i requisiti dell'assignment e ottimizzare ulteriormente il codice.

---

## 🖼️ 1. Allineamento Requisiti Grafici (Required Plots)
L'assignment PDF richiede esplicitamente alcuni grafici che mancavano o venivano sovrascritti al runtime in V5. In V6 sono stati aggiunti e configurati per il salvataggio automatico in PNG:

*   **Traiettorie Intermedie (Newton swing-up)**:
    *   *Modifica*: In `task1_main.py` e `task2_main.py` è stato aggiunto un grafico che visualizza su un unico asse temporale le traiettorie degli angoli ($\theta_1, \theta_2$) ad iterazioni intermedie significative (Iter 0/Warm Start, Iter 1, Iter 3, e Iter Ottima/Converged).
    *   *Output*: Salvato in automatico come `task1_intermediate_trajectories.png` e `task2_intermediate_trajectories.png`.
*   **Backtracking di Armijo per Iterazioni Iniziali e Finali**:
    *   *Modifica*: In `armijo.py` è stata modificata la firma di `select_stepsize` per accettare un parametro `save_path`. Nelle iterazioni critiche (k=0, k=1, e all'ultima iterazione convergente), i main passano un nome di file PNG univoco.
    *   *Output*: Salvati automaticamente come `task1_armijo_iter_0.png`, `task1_armijo_iter_1.png`, `task1_armijo_iter_X.png` (e corrispondenti `task2_armijo_iter_*.png`).
*   **Salvataggio Automatico di tutti i Grafici (PNG)**:
    *   *Modifica*: Aggiunte chiamate `plt.savefig()` ad alta risoluzione (300 DPI) per salvare tutti i grafici di convergenza, traiettoria ottima, errori di tracking e risultati finali dei Task 1, 2, 3 e 4.
    *   *Vantaggio*: Lo studente può inserire direttamente i file PNG all'interno del report in LaTeX senza dover fare screenshot.

---

## ⚡ 2. Ottimizzazione delle Prestazioni (Dynamics step)
*   *Modifica*: Mantenuta ed estesa l'ottimizzazione che separa `dynamics.step` (propagazione RK4 pura, 11x più veloce) da `dynamics.dynamics` (Jacobiani numerici con perturbazioni centrali) per evitare che Python si blocchi su Windows.
*   *Vantaggio*: Il loop di Armijo e le simulazioni di tracking girano ora in meno di un secondo.

---

## 📚 3. Conformità Teorica e Sintassi del Professore
*   **Sistema KKT e SQP (Equilibri)**:
    *   Verificato che in `equilibrium_finding.py` il sistema KKT sia impostato esattamente come in Slide 03:
        $$\begin{bmatrix} B_k & \partial h^T \\ \partial h & 0 \end{bmatrix} \begin{bmatrix} \Delta z \\ \lambda \end{bmatrix} = \begin{bmatrix} -\partial c \\ -h \end{bmatrix}$$
        con Hessiana del costo $B_k = \text{block\_diag}(Q, R)$ e vincolo $h(x,u) = F(x,u) - x = 0$.
*   **iDDP e Newton (Task 1 & 2)**:
    *   Confermato che in `solver_newton.py` la propagazione della Value Function usi la formula di Riccati discretizzata in closed-loop, con la corretta espansione al secondo ordine ($Q_x, Q_u, Q_{xx}, Q_{uu}, Q_{ux}$) presentata a lezione (Slide 08).
*   **DARE per Costo Terminale**:
    *   `task1_main.py` e `task2_main.py` utilizzano la funzione `control.dare` all'equilibrio obiettivo instabile (pendolo eretto) per calcolare la matrice $Q_T$ infinito-orizzonte in accordo con gli esempi in classe (`CodingSession4/10_main_...`).
*   **TV-LQR e MPC (Task 3 & 4)**:
    *   `solver_ltv_lqr.py` implementa l'equazione di Riccati discreta tempo-variante nella forma numerica standard raccomandata (FORM 1, conforme a `CodingSession5`), mentre `task4_main.py` formula il problema MPC in variabili di deviazione $\delta x_t, \delta u_t$ con vincoli sull'ingresso assoluto e terminal cost ereditato da Riccati.
