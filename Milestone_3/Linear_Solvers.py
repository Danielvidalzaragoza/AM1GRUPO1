import numpy as np
from numpy.linalg import norm
from numpy import array
from typing import Callable

def My_Gauss_Seidel(F:Callable[[array,float], array], U:array, At:float, t:float):
    # Método de Gauss-Seidel para resolver ecuaciones no lineales.
    Y = U + At * F(U, t)
    while True:
        Y_new = U + At * F(Y, t + At)
        if norm(Y_new - Y) < 1e-10:
            return Y_new
        Y = Y_new
