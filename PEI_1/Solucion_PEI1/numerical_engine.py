"""Esquemas temporales y estimacion de error para el problema de Cauchy."""

import numpy as np
from numpy.linalg import norm

if __package__:
    from .space_physics import estabilidad
else:
    from space_physics import estabilidad


def EE(U, At, t, F):
    return U + At * F(U, t)


def EI(U, At, t, F, J):
    return My_Newton(U, At, t, F, J)


def CN(U, At, t, F, J):
    def G(X, time):
        return (F(U, t) + F(X, time)) / 2

    def JG(X, time):
        return J(X, time) / 2

    return My_Newton(U, At, t, G, JG)


def RK4(U, At, t, F):
    k1 = F(U, t)
    k2 = F(U + At * k1 / 2, t + At / 2)
    k3 = F(U + At * k2 / 2, t + At / 2)
    k4 = F(U + At * k3, t + At)
    return U + At * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def LF(U, At, t, F, U_prev):
    return U_prev + 2 * At * F(U, t)


def My_Newton(U, At, t, F, J, tol=1e-10):
    Y = U + At * F(U, t)
    R = Y - U - At * F(Y, t + At)
    while norm(R) > tol:
        matrix = np.eye(len(U)) - At * J(Y, t + At)
        Y = Y - np.linalg.solve(matrix, R)
        R = Y - U - At * F(Y, t + At)
    return Y


def paso_Cauchy(esquema, U, U_prev, At, t, F, J):
    metodos = {"EE": EE, "EI": EI, "CN": CN, "RK4": RK4, "LF": LF}
    if esquema == "LF":
        return LF(U, At, t, F, U_prev)
    if esquema in ("EI", "CN"):
        return metodos[esquema](U, At, t, F, J)
    return metodos[esquema](U, At, t, F)


def integrar_Cauchy(esquema, U0, N_t, t0, tf, F, J):
    U = np.zeros((N_t + 1, len(U0)))
    t = np.linspace(t0, tf, N_t + 1)
    U[0] = U0
    At = t[1] - t[0]

    if esquema == "LF":
        U[1] = RK4(U[0], At, t[0], F)
        for n in range(1, N_t):
            U[n + 1] = paso_Cauchy(
                esquema, U[n], U[n - 1], At, t[n], F, J
            )
    else:
        for n in range(N_t):
            U[n + 1] = paso_Cauchy(esquema, U[n], None, At, t[n], F, J)
    return t, U


def richardson_error(esquema, U0, N_t, t0, tf, F, J):
    t, coarse = integrar_Cauchy(esquema, U0, N_t, t0, tf, F, J)
    _, fine = integrar_Cauchy(esquema, U0, 2 * N_t, t0, tf, F, J)
    _, finer = integrar_Cauchy(esquema, U0, 4 * N_t, t0, tf, F, J)

    difference_1 = norm(coarse - fine[::2])
    difference_2 = norm(fine[::2] - finer[::4])
    order = np.log(difference_1 / difference_2) / np.log(2)
    error = (coarse - fine[::2]) / (2**order - 1)
    return t, coarse, error, order


def stability_report(esquema, eigenvalues, At):
    z = At * np.asarray(eigenvalues)
    if esquema == "LF":
        roots = np.stack((z + np.sqrt(1 + z**2), z - np.sqrt(1 + z**2)))
        maximum = np.max(np.abs(roots))
    else:
        maximum = np.max(np.abs(estabilidad(esquema, z)))
    return maximum <= 1, maximum
