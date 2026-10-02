#Modelo Físico
import numpy as np
from numpy import array, zeros
from numpy.linalg import norm
from typing import Callable

def EE(U:array ,At:float,t:float, F:Callable[[array,float], array]):
	
	return U+ At * F(U,t)

def EI(U:array, At:float, t:float, F:Callable[[array,float], array]):
   # Método de Euler implícito para un solo paso: U(t+At) = U(t) + F(U(t+At), t+At)
   Y = U + At* F(U, t)
   while norm(Y - U - At* F(Y,t+At)) > 1e-10:
      R = Y - U - At * F(Y,t+At)
      Y = Y - R
   return Y


def CN(U:array, At:float, t:float, F:Callable[[array,float], array]):
    # Método de Crank-Nicolson de un solo paso: U(t+At) = U(t) + At/2 * {F(U(t), t) + F(U(t+At), t+At)}.
    Y = U + At * F(U, t)
    while norm(Y - U - At/2 * (F(U, t) + F(Y, t + At))) > 1e-10:
        R = Y - U - At/2 * (F(U, t) + F(Y, t + At))
        Y = Y - R
    return Y

def RK4(U:array, At:float, t:float, F:Callable[[array,float], array]):
    # Esquema básico de Runge-Kutta de cuarto orden.
    k1 = F(U,t)
    k2 = F(U, t + At * k1 / 2)
    k3 = F(U, t + At * k2 / 2)
    k4 = F(U, t + At * k3)   
    return U + At * (k1 + 2 * k2 + 2 * k3 + k4) / 6
