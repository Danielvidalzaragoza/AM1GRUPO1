import numpy as np
from typing import Callable

from Esquemas_temporales import CN, EE, EI, RK4


def partition(t0: float, tf: float, N: float):
    # Crea una partición del intervalo [t0, tf] en N subintervalos equiespaciados.
    return np.array([t0 + i * (tf - t0) / N for i in range(N + 1)])


def integrar_Cauchy(
    esquema: str,
    U0: np.ndarray,
    N_t: int,
    t0: float,
    tf: float,
    F: Callable[[np.ndarray, float], np.ndarray],
):
    metodos = {"EE": EE, "EI": EI, "CN": CN, "RK4": RK4}
    U = np.zeros((N_t + 1, len(U0)))
    U[0, :] = U0
    t = partition(t0, tf, N_t)
    metodo = metodos[esquema]
    for n in range(N_t):
        U[n + 1, :] = metodo(U[n, :], t[n + 1] - t[n], t[n], F)
    return U
