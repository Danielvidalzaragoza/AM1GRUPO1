import numpy as np
import Integradores
import Richardson
from typing import Callable

def sol_error_cauchy(
    esquema: str,
    U0: np.ndarray,
    N_t: int,
    t0: float,
    tf: float,
    F: Callable[[np.ndarray, float], np.ndarray],
):
    coarse = Integradores.integrar_Cauchy(esquema, U0, N_t, t0, tf, F)
    fine = Integradores.integrar_Cauchy(esquema, U0, 2 * N_t, t0, tf, F)
    finer = Integradores.integrar_Cauchy(esquema, U0, 4 * N_t, t0, tf, F)

    difference_coarse_fine = np.linalg.norm(coarse - fine[::2])
    difference_fine_finer = np.linalg.norm(fine[::2] - finer[::4])
    orden = np.log(difference_coarse_fine / difference_fine_finer) / np.log(2)
    error = Richardson.error_richardson(coarse, fine[::2], orden)
    return coarse, error, orden
