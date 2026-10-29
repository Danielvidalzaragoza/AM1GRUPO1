import numpy as np
from numpy.linalg import norm
from typing import Callable


def My_Newton(
    U: np.ndarray,
    At: float,
    t: float,
    F: Callable[[np.ndarray, float], np.ndarray],
    J: Callable[[np.ndarray, float], np.ndarray] = None,
    tol: float = 1e-10,
    max_iter: int = 50,
):
    Y = U + At * F(U, t)
    for _ in range(max_iter):
        R = Y - U - At * F(Y, t + At)
        if norm(R) <= tol:
            return Y
        jacobian = J(Y, t + At) if J is not None else My_Jacobian(F, Y, t + At)
        matrix = np.eye(len(U)) - At * jacobian
        Y = Y - np.linalg.solve(matrix, R)

    raise RuntimeError(f"Newton no convergió en {max_iter} iteraciones.")


def My_Jacobian(
    F: Callable[[np.ndarray, float], np.ndarray],
    U: np.ndarray,
    t: float,
) -> np.ndarray:
    """Aproxima el Jacobiano con diferencias finitas centradas de segundo orden."""
    n = len(U)
    jacobian = np.zeros((n, n))
    paso_relativo = np.cbrt(np.finfo(float).eps)
    for i in range(n):
        h = paso_relativo * max(1.0, abs(U[i]))
        U_mas = U.copy()
        U_menos = U.copy()
        U_mas[i] += h
        U_menos[i] -= h
        jacobian[:, i] = (F(U_mas, t) - F(U_menos, t)) / (2 * h)
    return jacobian
