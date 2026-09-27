from numpy import array, zeros

def Cauchy_problem(Esquema, U0: array, F, t: array) -> array:
    N = len(t) - 1
    Nv = len(U0)

    U = zeros((N + 1, Nv))
    U[0, :] = U0

    for n in range(N):
        Dt = t[n + 1] - t[n]
        U[n + 1, :] = Esquema(U[n, :], t[n], Dt, F)

    return U