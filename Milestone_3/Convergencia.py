import numpy as np
import Integradores


def converge_sys(esquema: str, U0: np.ndarray, t_i: float, t_f: float, F):
    """Return final-time coarse/fine differences and estimate their convergence order."""
    pasos = np.logspace(0, 5, num=10, dtype=int)
    error_final = np.zeros(len(pasos))
    for indice, n in enumerate(pasos):
        coarse = Integradores.integrar_Cauchy(esquema, U0, n, t_i, t_f, F)
        fine = Integradores.integrar_Cauchy(esquema, U0, 2 * n, t_i, t_f, F)
        error_final[indice] = np.linalg.norm(coarse[-1] - fine[-1])

    escala = np.maximum(np.linalg.norm(U0), 1.0)
    limite_redondeo = 10 * np.finfo(float).eps * pasos * escala
    ajuste = (pasos >= 20) & (error_final > limite_redondeo)

    pendiente, _ = np.polyfit(
        np.log(pasos[ajuste]), np.log(error_final[ajuste]), 1
    )
    orden = -pendiente
    print(
        f"Esquema: {esquema}, Orden estimado: {orden:.2f} "
        f"(ajuste con N >= {pasos[ajuste][0]})"
    )
    return error_final, pasos
