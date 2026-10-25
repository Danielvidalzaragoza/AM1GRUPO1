"""Esquemas temporales y estimación de error para el problema de Cauchy."""

import numpy as np
from numpy.linalg import norm
from typing import Callable

if __package__:
    from .space_physics import estabilidad
else:
    from space_physics import estabilidad


def EE(U: np.ndarray, At: float, t: float, F: Callable[[np.ndarray, float], np.ndarray]):
    return U + At * F(U, t)


def EI(U: np.ndarray, At: float, t: float, F: Callable[[np.ndarray, float], np.ndarray], J: Callable[[np.ndarray, float], np.ndarray] = None):
    Y = U + At * F(U, t)
    for _ in range(50):
        R = Y - U - At * F(Y, t + At)
        if norm(R) <= 1e-10:
            return Y
        if J is not None:
            mat_J = np.eye(len(U)) - At * J(Y, t + At)
        else:
            n = len(U)
            mat_J = np.eye(n)
            h_jac = 1e-8
            for i in range(n):
                U_plus = Y.copy()
                U_plus[i] += h_jac
                mat_J[:, i] -= At * (F(U_plus, t + At) - F(Y, t + At)) / h_jac
        Y = Y - np.linalg.solve(mat_J, R)
    return Y


def CN(U: np.ndarray, At: float, t: float, F: Callable[[np.ndarray, float], np.ndarray], J: Callable[[np.ndarray, float], np.ndarray] = None):
    def G(X: np.ndarray, time: float):
        return (F(U, t) + F(X, time)) / 2
    def JG(X: np.ndarray, time: float):
        if J is not None:
            return J(X, time) / 2
        return None
    return EI(U, At, t, G, JG)


def RK4(U: np.ndarray, At: float, t: float, F: Callable[[np.ndarray, float], np.ndarray]):
    k1 = F(U, t)
    k2 = F(U + At * k1 / 2, t + At / 2)
    k3 = F(U + At * k2 / 2, t + At / 2)
    k4 = F(U + At * k3, t + At)
    return U + At * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def LF(U: np.ndarray, At: float, t: float, F: Callable[[np.ndarray, float], np.ndarray], U_prev: np.ndarray):
    return U_prev + 2 * At * F(U, t)


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


def converge_sys(esquema: str, U0: np.ndarray, t_i: float, t_f: float, F, J=None):
    pasos = np.logspace(0, 5, num=10, dtype=int)
    error_final = np.zeros(len(pasos))
    for indice, n in enumerate(pasos):
        _, coarse = integrar_Cauchy(esquema, U0, int(n), t_i, t_f, F, J)
        _, fine = integrar_Cauchy(esquema, U0, int(2 * n), t_i, t_f, F, J)
        error_final[indice] = norm(coarse[-1] - fine[-1])

    escala = max(norm(U0), 1.0)
    limite_redondeo = 10 * np.finfo(float).eps * pasos * escala
    ajuste = (pasos >= 20) & (error_final > limite_redondeo)

    if np.sum(ajuste) >= 2:
        pendiente, _ = np.polyfit(np.log(pasos[ajuste]), np.log(error_final[ajuste]), 1)
        orden = -pendiente
    else:
        orden = 1.0
    return orden



def datos_convergencia(esquema, U0, t0, tf, F, J=None):

    numeros_pasos = np.array([20, 40, 80, 160, 320])
    numeros_nodos = []
    errores = []
    for N in numeros_pasos:
        _, solucion_gruesa = integrar_Cauchy(
            esquema, U0, int(N), t0, tf, F, J
        )
        _, solucion_fina = integrar_Cauchy(
            esquema, U0, int(2 * N), t0, tf, F, J
        )
        error = norm(
            solucion_gruesa[-1] - solucion_fina[-1]
        )
        numeros_nodos.append(N + 1)
        errores.append(error)
    return np.array(numeros_nodos), np.array(errores)

def richardson_error(
    esquema: str,
    U0: np.ndarray,
    N_t: int,
    t0: float,
    tf: float,
    F: Callable[[np.ndarray, float], np.ndarray],
    J: Callable[[np.ndarray, float], np.ndarray] = None,
):
    t_coarse, coarse = integrar_Cauchy(esquema, U0, N_t, t0, tf, F, J)
    _, fine = integrar_Cauchy(esquema, U0, 2 * N_t, t0, tf, F, J)
    _, finer = integrar_Cauchy(esquema, U0, 4 * N_t, t0, tf, F, J)

    difference_coarse_fine = norm(coarse - fine[::2])
    difference_fine_finer = norm(fine[::2] - finer[::4])
    
    if difference_fine_finer > 1e-15:
        orden = np.log(max(difference_coarse_fine, 1e-16) / difference_fine_finer) / np.log(2)
    else:
        orden = converge_sys(esquema, U0, t0, tf, F, J)

    min_len = min(len(coarse), len(fine[::2]))
    error = np.zeros_like(coarse)
    factor = 2**orden - 1
    if abs(factor) > 1e-8:
        error[:min_len] = (coarse[:min_len] - fine[::2][:min_len]) / factor
    else:
        error[:min_len] = coarse[:min_len] - fine[::2][:min_len]

    return t_coarse, coarse, error, float(orden)


def stability_report(esquema: str, eigenvalues: np.ndarray, At: float):
    z = At * np.asarray(eigenvalues)
    if esquema == "LF":
        roots = np.stack((z + np.sqrt(1 + z**2), z - np.sqrt(1 + z**2)))
        maximum = np.max(np.abs(roots))
    else:
        maximum = np.max(np.abs(estabilidad(esquema, z)))
    return maximum <= 1, maximum


def obtener_region_estabilidad(esquema: str, resolucion: int = 300):
    x = np.linspace(-4.0, 2.0, resolucion)
    y = np.linspace(-3.0, 3.0, resolucion)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    R = estabilidad(esquema, Z)
    return X, Y, np.abs(R)


def obtener_todas_regiones_estabilidad(esquemas=("EE", "EI", "CN", "RK4", "LF"), resolucion=200):
    x = np.linspace(-4.0, 2.0, resolucion)
    y = np.linspace(-3.0, 3.0, resolucion)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    regiones = {}
    for esq in esquemas:
        R = estabilidad(esq, Z)
        regiones[esq] = np.abs(R)
    return X, Y, regiones