import numpy as np
from numpy import array
from numpy.linalg import norm
from typing import Callable

from Milestone_3 import Integradores

def partition(t0: float, tf: float, N: float):
    # Crea una partición del intervalo [t0, tf] en N subintervalos equiespaciados.
    return np.array([t0 + i * (tf - t0) / N for i in range(N + 1)])

def sol_error_cauchy(
    esquema: str,
    U0: np.ndarray,
    N_t: int,
    t0: float,
    tf: float,
    F: Callable[[np.ndarray, float], np.ndarray],
):
    coarse = integrar_Cauchy(esquema, U0, N_t, t0, tf, F)
    fine = integrar_Cauchy(esquema, U0, 2 * N_t, t0, tf, F)
    finer = integrar_Cauchy(esquema, U0, 4 * N_t, t0, tf, F)

    difference_coarse_fine = np.linalg.norm(coarse - fine[::2])
    difference_fine_finer = np.linalg.norm(fine[::2] - finer[::4])
    orden = np.log(difference_coarse_fine / difference_fine_finer) / np.log(2)
    error = error_richardson(coarse, fine[::2], orden)
    return coarse, error, orden


def integrar_Cauchy(
    esquema: str,
    U0: np.ndarray,
    N_t: int,
    t0: float,
    tf: float,
    F: Callable[[np.ndarray, float], np.ndarray],
):
    metodos = {"EE": EE, "EI": EI, "CN": CN, "RK4": RK4, "LF": LF}
    U = np.zeros((N_t + 1, len(U0)))
    U[0, :] = U0
    t = partition(t0, tf, N_t)
    metodo = metodos[esquema]
    if esquema == "LF":
        # Para el esquema Leapfrog, necesitamos un paso inicial con otro método.
        U[1, :] = EE(U[0, :], t[1] - t[0], t[0], F)
        for n in range(1, N_t):
            U[n + 1, :] = metodo(U[n, :], t[n + 1] - t[n], t[n], F, U[n - 1, :])
    else:
        for n in range(N_t):
            U[n + 1, :] = metodo(U[n, :], t[n + 1] - t[n], t[n], F)
    return U


''' Cálculo del error temporal mediante extrapolación de Richardson.
'''
def error_richardson(
    coarse: np.ndarray,
    fine: np.ndarray,
    order: float,
) -> np.ndarray:
    """
    Estima el error de la aproximación fina mediante extrapolación de Richardson.
    `coarse` y `fine` deben representar la misma magnitud en los mismos tiempos,
    calculada con pasos h y h/n, respectivamente.
    """
    return (coarse - fine) / (2**order - 1)

''' Esquemas temporales para la integración de ecuaciones diferenciales ordinarias (EDO)
    -Euler Explícito- EE
    -Euler Implícito- EI
    -Crank-Nicolson- CN
    -Runge-Kutta de 4to orden- RK4
'''
def EE(U:array, At:float, t:float, F:Callable[[array,float], array]):
    return U + At * F(U, t)

def EI(U:array, At:float, t:float, F:Callable[[array,float], array]):
    # Método de Euler implícito mediante Newton.
    return My_Newton(U, At, t, F)


def CN(U:array, At:float, t:float, F:Callable[[array,float], array]):
    # Método de Crank-Nicolson mediante Newton.
    def G(X:array, time:float):
        return (F(U, t) + F(X, time)) / 2
    return My_Newton(U, At, t, G)


def RK4(U:array, At:float, t:float, F:Callable[[array,float], array]):
    # Esquema básico de Runge-Kutta de cuarto orden.
    k1 = F(U, t)
    k2 = F(U + At * k1 / 2, t + At / 2)
    k3 = F(U + At * k2 / 2, t + At / 2)
    k4 = F(U + At * k3, t + At)
    return U + At * (k1 + 2 * k2 + 2 * k3 + k4) / 6

def LF(U:array, At:float, t:float, F:Callable[[array,float], array], U_prev:array):
    # Esquema Leapfrog (salto de rana) de segundo orden.
    return U_prev + 2 * At * F(U, t)


def My_Newton(U:array, At:float, t:float, F:Callable[[array,float], array],
              tol:float=1e-10, max_iter:int=50):
    # Método de Newton para resolver ecuaciones no lineales.
    Y = U + At * F(U, t)
    for _ in range(max_iter):
        R = Y - U - At * F(Y, t + At)
        if norm(R) <= tol:
            return Y
        J = np.eye(len(U)) - At * My_Jacobian(F, Y, t + At)
        Y = Y - np.linalg.solve(J, R)

    raise RuntimeError(f"Newton no convergió en {max_iter} iteraciones.")


def My_Gauss_Seidel(F:Callable[[array,float], array], U:array, At:float, t:float):
    # Método de Gauss-Seidel para resolver ecuaciones no lineales.
    Y = U + At * F(U, t)
    while True:
        Y_new = U + At * F(Y, t + At)
        if norm(Y_new - Y) < 1e-10:
            return Y_new
        Y = Y_new


def My_Jacobian(F:Callable[[array,float], array], U:array, t:float):
    # Aproximación de la matriz jacobiana de F en el punto U.
    n = len(U)
    J = np.zeros((n, n))
    h = 1e-8
    for i in range(n):
        U_plus = U.copy()
        U_plus[i] += h
        J[:, i] = (F(U_plus, t) - F(U, t)) / h
    return J
