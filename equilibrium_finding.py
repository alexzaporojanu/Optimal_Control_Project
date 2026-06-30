#
# Acrobot — Ricerca Equilibri via SQP
# Progetto Optimal Control — Parameter Set 3
#
# Riferimento teorico:
#   [Slide 03] KKT Conditions            — sezione "Equality Constrained Optimization"
#   [Slide 06] Optimal Control Shooting  — sezione "Equilibrium Points"
#   [Session3/5c_SQP_equilibrium_finding.py] — base dell'implementazione
#
# PROBLEMA DI EQUILIBRIO
# ======================
# Un equilibrio (x*, u*) è un punto fisso della dinamica discreta:
#
#   F(x*, u*) = x*   →   F(x*, u*) - x* = 0
#
# Questo è un sistema di 4 equazioni nonlineari in 5 incognite (z = [x; u]).
# Ha 1 grado di libertà (DoF) → non è determinato.
#
# Per risolverlo usiamo SQP (Sequential Quadratic Programming):
# ad ogni iterazione k, risolviamo il sottoproblema KKT:
#
#   [B_k   ∂h/∂z^T] [Δz  ]   [-∂c/∂z]
#   [∂h/∂z    0   ] [λ   ] = [-h(z_k)]
#
# dove:
#   h(z) = F(x,u) - x   ← vincolo di uguaglianza (4 equazioni)
#   c(z) = ½ (x-xg)ᵀ Q (x-xg) + ½ uᵀ R u  ← costo (rimane vicino a guess)
#   B_k  = Hessiana del Lagrangiano approssimata con block diag(Q, R)
#   λ    ← moltiplicatori di Lagrange (KKT)
#
# La soluzione Δz porta il punto corrente verso la feasibility del vincolo.
# [Rif.: Slide 03 — "KKT Conditions for Equality Constrained Problems"]
#

import numpy as np
import scipy.linalg
from dynamics import dynamics


def _equality_constraint(xx, uu):
    """
    Calcola il vincolo di equilibrio h(x,u) = F(x,u) - x e il suo Jacobiano.

    h(x, u) = F(x, u) - x   ∈ ℝ⁴

    ∂h/∂z = [∂F/∂x - I,  ∂F/∂u] = [A - I,  B]   ∈ ℝ⁴ˣ⁵

    dove A = ∂F/∂x, B = ∂F/∂u sono i Jacobiani della dinamica discreta.
    [Rif.: Slide 03 — "Constraint Jacobians"]
    """
    x_next, A, B = dynamics(xx, uu)

    h    = x_next - xx               # (4,) — violazione vincolo
    dh_x = A - np.eye(len(xx))       # (4×4)
    dh_u = B                          # (4×1)
    dh   = np.hstack([dh_x, dh_u])  # (4×5) — Jacobiano combinato

    return h, dh


def _cost_quadratic(xx, uu, Q, R, xref, uref):
    """
    Costo quadratico per mantenere la soluzione vicina alla guess iniziale.

    c(x, u)  = ½ (x-xref)ᵀ Q (x-xref) + ½ (u-uref)ᵀ R (u-uref)
    ∂c/∂z    = [Q(x-xref); R(u-uref)]   (vettore combinato)
    ∂²c/∂z²  = block_diag(Q, R)         (Hessiana = B_k nell'SQP)

    [Rif.: Slide 03 — "Augmented Lagrangian / SQP"]
    """
    dx = xx - xref
    du = uu - uref

    c    = 0.5 * dx.T @ Q @ dx + 0.5 * du.T @ R @ du
    dc   = np.hstack([Q @ dx, R @ du])           # (5,)
    ddc  = scipy.linalg.block_diag(Q, R)         # (5×5)

    return float(c), dc, ddc


def find_equilibrium(x_guess, u_guess, label="", max_iters=50, tol=1e-8):
    """
    Trova un equilibrio (x*, u*) dell'Acrobot con SQP.

    Risolve iterativamente il sistema KKT:
        [B  ∂h^T] [Δz] = [-∂c]
        [∂h   0 ] [λ ] = [-h ]

    dove B = ∂²c/∂z², ∂h = Jacobiano del vincolo.
    Il sistema KKT si ottiene dalle condizioni di ottimalità del Lagrangiano:
        L(z, λ) = c(z) + λᵀ h(z)
        ∂L/∂z = ∂c + ∂hᵀ λ = 0
        ∂L/∂λ = h(z) = 0

    [Rif.: Slide 03 — "KKT System / Newton for Equality Constrained NLP"]
    [Rif.: Session3/5c_SQP_equilibrium_finding.py]

    Args:
        x_guess  : ndarray (4,) — stato iniziale guess
        u_guess  : ndarray (1,) — ingresso iniziale guess
        label    : str          — nome dell'equilibrio (per stampa)
        max_iters: int          — iterazioni massime SQP
        tol      : float        — tolleranza sulla norma del vincolo ||h||

    Returns:
        x_eq : ndarray (4,) — stato di equilibrio trovato
        u_eq : ndarray (1,) — ingresso di equilibrio trovato
    """
    print(f"\n--- Ricerca Equilibrio: {label} ---")

    nx, nu = 4, 1
    nz = nx + nu   # 5 variabili di decisione [x; u]

    z = np.hstack([x_guess.flatten(), u_guess.flatten()])   # z₀

    # Pesi del costo di regolarizzazione (rimane vicino alla guess)
    Q_reg = np.diag([10.0, 10.0, 1.0, 1.0])
    R_reg = np.diag([0.1])

    for k in range(max_iters):
        xx_k = z[:nx]
        uu_k = z[nx:]

        # 1. Calcolo costo e suo gradiente/Hessiana
        _, dc, B_k = _cost_quadratic(xx_k, uu_k, Q_reg, R_reg,
                                      x_guess.flatten(), u_guess.flatten())

        # 2. Calcolo vincolo e suo Jacobiano
        h_k, dh_k = _equality_constraint(xx_k, uu_k)

        # Controllo convergenza
        constr_norm = np.linalg.norm(h_k)
        if k % 10 == 0:
            print(f"  Iter {k:3d}: ||h|| = {constr_norm:.3e}")
        if constr_norm < tol:
            print(f"  Convergenza! ||h|| = {constr_norm:.2e} (< {tol:.0e}) "
                  f"a iter {k}.")
            break

        # 3. Costruzione e soluzione del sistema KKT (SQP step)
        #    [B_k   dh^T] [Δz] = [-dc]
        #    [dh_k    0 ] [λ ] = [-h ]
        # Dimensioni: B_k (5×5), dh_k (4×5) → sistema (9×9)
        KKT = np.block([
            [B_k,      dh_k.T               ],   # (5×5), (5×4)
            [dh_k,     np.zeros((nx, nx))   ]    # (4×5), (4×4)
        ])                                        # totale: (9×9)

        rhs = np.hstack([-dc, -h_k])             # (9,)

        # Risoluzione del sistema KKT lineare
        try:
            sol = np.linalg.solve(KKT, rhs)
        except np.linalg.LinAlgError:
            print(f"  WARNING [iter {k}]: Sistema KKT singolare. "
                  "Uso pseudo-inversa come fallback.")
            sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]

        dz = sol[:nz]   # estrai solo Δz (i moltiplicatori λ non servono qui)

        # 4. Update con step unitario (Newton puro — converge quadraticamente vicino)
        z = z + dz

    x_eq = z[:nx]
    u_eq = z[nx:]

    # Verifica finale
    x_check, _, _ = dynamics(x_eq, u_eq)
    err = np.linalg.norm(x_check - x_eq)
    print(f"  Equilibrio: x = {x_eq.round(6)}")
    print(f"              u = {u_eq.round(6)}")
    print(f"  Verifica F(x*,u*)-x* = {err:.2e}")

    return x_eq, u_eq


# =============================================================================
# ESECUZIONE PRINCIPALE
# =============================================================================
if __name__ == "__main__":

    # ---- Equilibrio 1: Pendolo Giù (posizione di riposo) ----
    # Guess: tutte le variabili nulle (sistema a riposo con coppia nulla)
    x_down_guess = np.array([0.0, 0.0, 0.0, 0.0])
    u_down_guess = np.array([0.0])
    x_eq1, u_eq1 = find_equilibrium(x_down_guess, u_down_guess, "DOWNWARD (θ₁=0)")

    # ---- Equilibrio 2: Pendolo Su (posizione instabile) ----
    # Guess: θ₁=π (inverted), zero velocità, zero coppia
    # Nota: l'equilibrio invertito è instabile ma esiste per l'Acrobot ideale
    x_up_guess = np.array([np.pi, 0.0, 0.0, 0.0])
    u_up_guess = np.array([0.0])
    x_eq2, u_eq2 = find_equilibrium(x_up_guess, u_up_guess, "UPWARD (θ₁=π)")

    # Salvataggio per riuso negli altri task
    np.save('equilibrium_data.npy', {
        'x_eq1': x_eq1,
        'x_eq2': x_eq2,
        'u_eq1': u_eq1,
        'u_eq2': u_eq2
    })
    print("\nDati di equilibrio salvati in 'equilibrium_data.npy'")
    print(f"x_eq1 (giù):   {x_eq1.round(4)}")
    print(f"x_eq2 (su):    {x_eq2.round(4)}")
