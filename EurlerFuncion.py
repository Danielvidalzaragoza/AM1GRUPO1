from numpy import array
 
def Euler(U: array, t: float, Dt: float, F) -> array:
    return U + Dt * F(U, t)