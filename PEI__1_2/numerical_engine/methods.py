import numpy as np
from typing import Callable


def EE(U: np.ndarray, At: float, t: float, F: Callable[[np.ndarray, float], np.ndarray]):
    return U + At * F(U, t)

def EI(U: np.ndarray, At: float, t: float,
       F: Callable[[np.ndarray, float], np.ndarray], 
       J: Callable[[np.ndarray, float], np.ndarray] = None):
    from .solvers import My_Newton

    return My_Newton(U, At, t, F, J)

def CN(U: np.ndarray, At: float, t: float, 
       F: Callable[[np.ndarray, float], np.ndarray], 
       J: Callable[[np.ndarray, float], np.ndarray] = None):
    from .solvers import My_Newton

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
