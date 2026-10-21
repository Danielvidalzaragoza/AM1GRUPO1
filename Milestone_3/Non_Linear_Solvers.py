import numpy as np
from numpy.linalg import norm
from numpy import array
from typing import Callable

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
