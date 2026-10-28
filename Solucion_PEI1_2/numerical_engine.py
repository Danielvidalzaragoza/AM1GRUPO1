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


def EI(U: np.ndarray, At: float, t: float,
       F: Callable[[np.ndarray, float], np.ndarray], 
       J: Callable[[np.ndarray, float], np.ndarray] = None):
    return My_Newton(U, At, t, F, J)


def CN(U: np.ndarray, At: float, t: float, 
       F: Callable[[np.ndarray, float], np.ndarray], 
       J: Callable[[np.ndarray, float], np.ndarray] = None):
    def G(X: np.ndarray, time: float):
        return (F(U, t) + F(X, time)) / 2

    def JG(X: np.ndarray, time: float):
        return J(X, time) / 2

    return My_Newton(U, At, t, G, JG if J is not None else None)


def RK4(U: np.ndarray, At: float, t: float, F: Callable[[np.ndarray, float], np.ndarray]):
    k1 = F(U, t)
    k2 = F(U + At * k1 / 2, t + At / 2)
    k3 = F(U + At * k2 / 2, t + At / 2)
    k4 = F(U + At * k3, t + At)
    return U + At * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def LF(U: np.ndarray, At: float, t: float, F: Callable[[np.ndarray, float], np.ndarray], U_prev: np.ndarray):
    return U_prev + 2 * At * F(U, t)

def My_Newton(
    U: np.ndarray,
    At: float,
    t: float,
    F: Callable[[np.ndarray, float], np.ndarray],
    J: Callable[[np.ndarray, float], np.ndarray] = None,
    tol: float = 1e-10,
    max_iter: int = 50,
):
    # Método de Newton para resolver ecuaciones no lineales.
    Y = U + At * F(U, t)
    for _ in range(max_iter):
        R = Y - U - At * F(Y, t + At)
        if norm(R) <= tol:
            return Y
        jacobian = J(Y, t + At) if J is not None else My_Jacobian(F, Y, t + At)
        matrix = np.eye(len(U)) - At * jacobian
        Y = Y - np.linalg.solve(matrix, R)

    raise RuntimeError(f"Newton no convergió en {max_iter} iteraciones.")


def My_Jacobian(
    F: Callable[[np.ndarray, float], np.ndarray],
    U: np.ndarray,
    t: float,
) -> np.ndarray:
    """Aproxima por diferencias finitas el Jacobiano de F respecto a U."""
    n = len(U)
    jacobian = np.zeros((n, n))
    h = 1e-8
    for i in range(n):
        U_plus = U.copy()
        U_plus[i] += h
        jacobian[:, i] = (F(U_plus, t) - F(U, t)) / h
    return jacobian

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