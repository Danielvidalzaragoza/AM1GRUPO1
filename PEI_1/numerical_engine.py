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


def EI(U: np.ndarray, At: float, t: float, F: Callable[[np.ndarray, float], np.ndarray], J: Callable[[np.ndarray, float], np.ndarray]):
    return My_Newton(U, At, t, F, J)


def CN(U: np.ndarray, At: float, t: float, F: Callable[[np.ndarray, float], np.ndarray], J: Callable[[np.ndarray, float], np.ndarray]):
    def G(X: np.ndarray, time: float):
        return (F(U, t) + F(X, time)) / 2

    def JG(X: np.ndarray, time: float):
        return J(X, time) / 2

    return My_Newton(U, At, t, G, JG)


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
    J: Callable[[np.ndarray, float], np.ndarray],
    tol: float = 1e-10,
    max_iter: int = 50,
):
    Y = U + At * F(U, t)
    R = Y - U - At * F(Y, t + At)
    iteraciones = 0
    while norm(R) > tol:
        iteraciones += 1
        if iteraciones > max_iter:
            raise RuntimeError(f"Newton no convergió en {max_iter} iteraciones.")
        matrix = np.eye(len(U)) - At * J(Y, t + At)
        Y = Y - np.linalg.solve(matrix, R)
        R = Y - U - At * F(Y, t + At)
    return Y


def paso_Cauchy(
    esquema: str,
    U: np.ndarray,
    U_prev: np.ndarray,
    At: float,
    t: float,
    F: Callable[[np.ndarray, float], np.ndarray],
    J: Callable[[np.ndarray, float], np.ndarray],
):
    metodos = {"EE": EE, "EI": EI, "CN": CN, "RK4": RK4, "LF": LF}
    if esquema == "LF":
        return LF(U, At, t, F, U_prev)
    if esquema in ("EI", "CN"):
        return metodos[esquema](U, At, t, F, J)
    return metodos[esquema](U, At, t, F)


def integrar_Cauchy(
    esquema: str,
    U0: np.ndarray,
    N_t: int,
    t0: float,
    tf: float,
    F: Callable[[np.ndarray, float], np.ndarray],
    J: Callable[[np.ndarray, float], np.ndarray],
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


def richardson_error(
    esquema: str,
    U0: np.ndarray,
    N_t: int,
    t0: float,
    tf: float,
    F: Callable[[np.ndarray, float], np.ndarray],
    J: Callable[[np.ndarray, float], np.ndarray],
):
    t, coarse = integrar_Cauchy(esquema, U0, N_t, t0, tf, F, J)
    _, fine = integrar_Cauchy(esquema, U0, 2 * N_t, t0, tf, F, J)
    _, finer = integrar_Cauchy(esquema, U0, 4 * N_t, t0, tf, F, J)

    difference_1 = norm(coarse - fine[::2])
    difference_2 = norm(fine[::2] - finer[::4])
    order = np.log(difference_1 / difference_2) / np.log(2)
    error = (coarse - fine[::2]) / (2**order - 1)
    return t, coarse, error, order


def stability_report(esquema: str, eigenvalues: np.ndarray, At: float):
    z = At * np.asarray(eigenvalues)
    if esquema == "LF":
        roots = np.stack((z + np.sqrt(1 + z**2), z - np.sqrt(1 + z**2)))
        maximum = np.max(np.abs(roots))
    else:
        maximum = np.max(np.abs(estabilidad(esquema, z)))
    return maximum <= 1, maximum