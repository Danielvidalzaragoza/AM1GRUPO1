import numpy as np
from typing import Callable

def integrar_Cauchy(
    esquema: str,
    U0: np.ndarray,
    At: float,
    N: int,
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

    U = np.zeros((N + 1, len(U0)))
    U[0, :] = U0
    for n in range(0, N):
        U[n + 1, :] = metodo(U[n, :], At, n * At, F)
    return U
