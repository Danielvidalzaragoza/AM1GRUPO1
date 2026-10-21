'''Esquemas temporales para la integración de ecuaciones diferenciales ordinarias (EDO)
    -Euler Explícito- EE
    -Euler Implícito- EI
    -Crank-Nicolson- CN
    -Runge-Kutta de 4to orden- RK4

    También se calculan las funciones de apoyo de Newton, Gauss-Seidel y Jacobiano para los métodos implícitos.
'''
import numpy as np
import Non_Linear_Solvers
from numpy import array
from numpy.linalg import norm
from typing import Callable


def EE(U:array, At:float, t:float, F:Callable[[array,float], array]):
    return U + At * F(U, t)


def EI(U:array, At:float, t:float, F:Callable[[array,float], array]):
    # Método de Euler implícito mediante Newton.
    return Non_Linear_Solvers.My_Newton(U, At, t, F)


def CN(U:array, At:float, t:float, F:Callable[[array,float], array]):
    # Método de Crank-Nicolson mediante Newton.
    def G(X:array, time:float):
        return (F(U, t) + F(X, time)) / 2
    return Non_Linear_Solvers.My_Newton(U, At, t, G)


def RK4(U:array, At:float, t:float, F:Callable[[array,float], array]):
    # Esquema básico de Runge-Kutta de cuarto orden.
    k1 = F(U, t)
    k2 = F(U + At * k1 / 2, t + At / 2)
    k3 = F(U + At * k2 / 2, t + At / 2)
    k4 = F(U + At * k3, t + At)
    return U + At * (k1 + 2 * k2 + 2 * k3 + k4) / 6


