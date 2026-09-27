from numpy import array
from numpy.linalg import norm


def Inverse_Euler(U: array, t: float, Dt: float, F) -> array:
    Y = U
    R = Y - U - Dt * F(Y, t + Dt)
    while norm(R) > 1e-6:
        Y = Y - R
        R = Y - U - Dt * F(Y, t + Dt)
    return Y