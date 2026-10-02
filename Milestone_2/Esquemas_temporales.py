#Modelo Físico
import numpy as np
from numpy import array, zeros
from numpy.linalg import norm
from typing import Callable

def EE(U:array ,At:float,t:float, F:Callable[[array,float], array]):
	
	return U+ At * F(U,t)

#def EI(U:array, At:float, t:float, F:Callable[[array,float], array]):
        # Método de Euler implícito: U[n+1] = U[n] + At/2 (F(U[n]) + F(U[n+1])).
 #   for n in range(0, N):
         #   Y = U[n, :]
        #    while norm(Y - U[n, :] - At / 2 * (F(U[n, :]) + F(Y))) > 1e-10:
       #          R = Y - U[n, :] - At / 2 * (F(U[n, :]) + F(Y))
      #           Y = Y - R
     #       U[n + 1, :] = Y
  #  return U

#def CN(U:array, At:float, t:float, F:Callable[[array,float], array]):
    # Método de Crank-Nicolson: U[n+1] = U[n] + At/2 (F(U[n]) + F(U[n+1]))
 #   for n in range(0, N):
    #    Y = U[n, :].copy()
 #       while norm(Y - U[n, :] - At / 2 * (F(U[n, :]) + F(Y))) > 1e-10:
  #          R = Y - U[n, :] - At / 2 * (F(U[n, :]) + F(Y))
   #         Y = Y - R
    #    U[n + 1, :] = Y
  #  return U

def RK4(U:array, At:float, t:float, F:Callable[[array,float], array]):
    # Esquema básico de Runge-Kutta de cuarto orden.
    k1 = F(U,t)
    k2 = F(U, t + At * k1 / 2)
    k3 = F(U, t + At * k2 / 2)
    k4 = F(U, t + At * k3)   
    return U + At * (k1 + 2 * k2 + 2 * k3 + k4) / 6
