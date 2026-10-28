'''Esquemas temporales para la integración de ecuaciones diferenciales ordinarias (EDO)
    -Euler Explícito- EE
    -Euler Implícito- EI
    -Crank-Nicolson- CN
    -Runge-Kutta de 4to orden- RK4
'''
import numpy as np
from numpy.linalg import norm
from typing import Callable

def EE(U: np.ndarray, At: float, t: float, F: Callable[[np.ndarray, float], np.ndarray]):
	return U+ At * F(U,t)

def EI(U: np.ndarray, At: float, t: float, F: Callable[[np.ndarray, float], np.ndarray]):
   # Método de Euler implícito para un solo paso: U(t+At) = U(t) + F(U(t+At), t+At)
   Y = U + At* F(U, t)
   while norm(Y - U - At* F(Y,t+At)) > 1e-10:
      R = Y - U - At * F(Y,t+At)
      Y = Y - R
   return Y

def CN(U: np.ndarray, At: float, t: float, F: Callable[[np.ndarray, float], np.ndarray]):
    # Método de Crank-Nicolson de un solo paso: U(t+At) = U(t) + At/2 * {F(U(t), t) + F(U(t+At), t+At)}.
    Y = U + At * F(U, t)
    while norm(Y - U - At/2 * (F(U, t) + F(Y, t + At))) > 1e-10:
        R = Y - U - At/2 * (F(U, t) + F(Y, t + At))
        Y = Y - R
    return Y

def RK4(U: np.ndarray, At: float, t: float, F: Callable[[np.ndarray, float], np.ndarray]):
    # Esquema básico de Runge-Kutta de cuarto orden.
    k1 = F(U,t)
    k2 = F(U + At * k1 / 2, t + At / 2)
    k3 = F(U + At * k2 / 2, t + At / 2)
    k4 = F(U + At * k3, t + At)
    return U + At * (k1 + 2 * k2 + 2 * k3 + k4) / 6
