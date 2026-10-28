'''Esquemas temporales para la integración de ecuaciones diferenciales ordinarias (EDO)
    -Euler Explícito- EE
    -Euler Implícito- EI
    -Crank-Nicolson- CN
    -Runge-Kutta de 4to orden- RK4
    
    También se calculan las funciones de apoyo de Newton, Gauss-Seidel y Jacobiano para los métodos implícitos.
'''
import numpy as np
from numpy import array, zeros
from numpy.linalg import norm
from typing import Callable


def EE(U:array ,At:float,t:float, F:Callable[[array,float], array]):
	return U+ At * F(U,t)

def EI(U:array, At:float, t:float, F:Callable[[array,float], array]):
   # Método de Euler implícito para un solo paso: U(t+At) = U(t) + At * F(U(t+At), t+At)
   return My_Newton(U, At, t, F)

def CN(U:array, At:float, t:float, F:Callable[[array,float], array]):
    # Método de Crank-Nicolson de un solo paso: U(t+At) = U(t) + At/2 * {F(U(t), t) + F(U(t+At), t+At)}.
    def G(X:array, time:float):
        return (F(U, t) + F(X, time)) / 2

    return My_Newton(U, At, t, G)

def RK4(U:array, At:float, t:float, F:Callable[[array,float], array]):
    # Esquema básico de Runge-Kutta de cuarto orden.
    k1 = F(U,t)
    k2 = F(U, t + At * k1 / 2)
    k3 = F(U, t + At * k2 / 2)
    k4 = F(U, t + At * k3)
    return U + At * (k1 + 2 * k2 + 2 * k3 + k4) / 6

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
    while norm(Y_new - Y) < 1e-10:
        Y_new = U + At * F(Y, t + At)
    Y = Y_new
    return Y

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