import numpy as np
from typing import Callable

from .methods import EE, EI, CN, RK4, LF


def paso_Cauchy(
    esquema: str,
    U: np.ndarray,
    U_prev: np.ndarray,
    At: float,
    t: float,
    F: Callable[[np.ndarray, float], np.ndarray],
    J: Callable[[np.ndarray, float], np.ndarray] = None,
):
    if esquema == "EE":
        return EE(U, At, t, F)
    elif esquema == "EI":
        return EI(U, At, t, F, J)
    elif esquema == "CN":
        return CN(U, At, t, F, J)
    elif esquema == "RK4":
        return RK4(U, At, t, F)
    elif esquema == "LF":
        return LF(U, At, t, F, U_prev)
    return EE(U, At, t, F)


def integrar_Cauchy(
    esquema: str,
    U0: np.ndarray,
    N_t: int,
    t0: float,
    tf: float,
    F: Callable[[np.ndarray, float], np.ndarray],
    J: Callable[[np.ndarray, float], np.ndarray] = None,
):
    U = np.zeros((N_t + 1, len(U0)))
    t = np.linspace(t0, tf, N_t + 1)
    U[0] = U0
    At = t[1] - t[0]

    if esquema == "LF":
        U[1] = RK4(U[0], At, t[0], F)
        for n in range(1, N_t):
            U[n + 1] = paso_Cauchy(esquema, U[n], U[n - 1], At, t[n], F, J)
    else:
        for n in range(N_t):
            U[n + 1] = paso_Cauchy(esquema, U[n], None, At, t[n], F, J)
    return t, U
