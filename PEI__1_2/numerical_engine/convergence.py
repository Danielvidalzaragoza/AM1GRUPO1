import numpy as np
from numpy.linalg import norm
from typing import Callable

from .integrators import integrar_Cauchy

def converge_sys(esquema: str, U0: np.ndarray, t_i: float, t_f: float, F, J=None):
    pasos = np.geomspace(20, 10_240, num=10, dtype=int)
    error_final = np.zeros(len(pasos))
    for indice, n in enumerate(pasos):
        _, coarse = integrar_Cauchy(esquema, U0, int(n), t_i, t_f, F, J)
        _, fine = integrar_Cauchy(esquema, U0, int(2 * n), t_i, t_f, F, J)
        error_final[indice] = norm(coarse[n] - fine[2*n])

    escala = max(norm(U0), 1.0)
    limite_redondeo = 10 * np.finfo(float).eps * pasos * escala
    ajuste = (pasos >= 20) & (error_final > limite_redondeo)

    if np.sum(ajuste) >= 2:
        pendiente, _ = np.polyfit(np.log(pasos[ajuste]), np.log(error_final[ajuste]), 1)
        orden = -pendiente
    else:
        orden = 1.0
    return orden, pasos, error_final

def richardson_error(esquema: str,U0: np.ndarray,
    N_t: int,t0: float,tf: float,
    F: Callable[[np.ndarray, float], np.ndarray],
    J: Callable[[np.ndarray, float], np.ndarray] = None,
):
    _, coarse = integrar_Cauchy(esquema, U0, N_t, t0, tf, F, J)
    _, fine = integrar_Cauchy(esquema, U0, 2 * N_t, t0, tf, F, J)
    _, finer = integrar_Cauchy(esquema, U0, 4 * N_t, t0, tf, F, J)

    difference_coarse_fine = norm(coarse - fine[::2])
    difference_fine_finer = norm(fine[::2] - finer[::4])
    
    if difference_fine_finer > 1e-15:
        orden = np.log(max(difference_coarse_fine, 1e-16) / difference_fine_finer) / np.log(2)
    else:
        orden, _, _ = converge_sys(esquema, U0, t0, tf, F, J)

    min_len = len(coarse)
    error = np.zeros_like(coarse)
    factor = 2**orden - 1
    error[:len(coarse)] = (coarse[:len(coarse)] - fine[::2][:len(coarse)]) / factor
   
    return  error, float(orden)
