from numpy import array
from typing import Callable

def Euler(U: array, t: float, Dt: float, F: Callable [[array, float], array]) -> array:
    
    return U + Dt * F(U, t)

def RK4(U: array, t: float, Dt: float, F) -> array:
    k_1 = F(U, t)
    k_2 = F(U + Dt * k_1 / 2, t + Dt / 2)
    k_3 = F(U + Dt * k_2 / 2, t + Dt / 2)
    k_4 = F(U + Dt * k_3, t + Dt)
    return U + Dt * (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6
