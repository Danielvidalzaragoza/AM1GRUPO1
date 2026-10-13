import numpy as np
from typing import Callable

def partition(t0:float, tf:float, N:float):
    # Crea una partición del intervalo [t0, tf] en N subintervalos equiespaciados.
    return np.array([t0 + i * (tf - t0) / N for i in range(N + 1)])

def integrar_Cauchy(
    esquema: str,
    U0: np.ndarray,
    N_t: int, # Numero de pasos de tiempo
    t0: float,
    tf: float,
    F: Callable[[np.ndarray, float], np.ndarray],
):
    if esquema == "EE":
        from Esquemas_temporales import EE as metodo    
    elif esquema == "EI":
        from Esquemas_temporales import EI as metodo
    elif esquema == "CN":
        from Esquemas_temporales import CN as metodo    
    elif esquema == "RK4":
        from Esquemas_temporales import RK4 as metodo

    U = np.zeros((N_t + 1, len(U0)))
    U[0, :] = U0
    t = partition(t0, tf, N_t)
    for n in range(0, N_t):
        U[n + 1, :] = metodo(U[n, :], t[n + 1] - t[n], t[n], F)
    return U
