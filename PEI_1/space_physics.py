"""Ecuaciones de Euler e integrales primeras del sólido rígido libre."""

import numpy as np


def ecuaciones_rotacion(W: np.ndarray, I1: float, I2: float, I3: float) -> np.ndarray:
    return np.array(
        [
            (I2 - I3) / I1 * W[1] * W[2],
            (I3 - I1) / I2 * W[2] * W[0],
            (I1 - I2) / I3 * W[0] * W[1],
        ]
    )


def jacobiano(W: np.ndarray, I1: float, I2: float, I3: float) -> np.ndarray:
    w1, w2, w3 = W
    return np.array(
        [
            [0, (I2 - I3) / I1 * w3, (I2 - I3) / I1 * w2],
            [(I3 - I1) / I2 * w3, 0, (I3 - I1) / I2 * w1],
            [(I1 - I2) / I3 * w2, (I1 - I2) / I3 * w1, 0],
        ]
    )


def invariantes(W: np.ndarray, I1: float, I2: float, I3: float):
    energia = 0.5 * (I1 * W[0] ** 2 + I2 * W[1] ** 2 + I3 * W[2] ** 2)
    momento = (I1 * W[0]) ** 2 + (I2 * W[1]) ** 2 + (I3 * W[2]) ** 2
    return energia, momento


def historial_invariantes(U: np.ndarray, I1: float, I2: float, I3: float):
    energia = 0.5 * (I1 * U[:, 0] ** 2 + I2 * U[:, 1] ** 2 + I3 * U[:, 2] ** 2)
    momento = (I1 * U[:, 0]) ** 2 + (I2 * U[:, 1]) ** 2 + (I3 * U[:, 2]) ** 2
    return energia, momento


def estabilidad_ejes(I1: float, I2: float, I3: float, omega: float = 1):
    equilibrios = {
        "I1": np.array([omega, 0, 0]),
        "I2": np.array([0, omega, 0]),
        "I3": np.array([0, 0, omega]),
    }
    jacobianos = {
        nombre: jacobiano(W, I1, I2, I3)
        for nombre, W in equilibrios.items()
    }
    autovalores = {
        nombre: np.linalg.eigvals(J)
        for nombre, J in jacobianos.items()
    }
    return equilibrios, jacobianos, autovalores


def separatriz(I1: float, I2: float, I3: float, energia: float, puntos: int = 160):
    pendiente = np.sqrt(I1 * (I2 - I1) / (I3 * (I3 - I2)))
    w1_max = np.sqrt(2 * energia / (I1 + I3 * pendiente**2))
    w2_max = np.sqrt(2 * energia / I2)
    phi = np.linspace(0, 2 * np.pi, puntos)
    w1 = w1_max * np.cos(phi)
    return np.column_stack((w1, w2_max * np.sin(phi), pendiente * w1))


def paso_actitud(R: np.ndarray, W: np.ndarray, W_nuevo: np.ndarray, dt: float) -> np.ndarray:
    w = (W + W_nuevo) / 2
    modulo = np.linalg.norm(w)
    if modulo == 0:
        return R
    theta = modulo * dt
    wx, wy, wz = w / modulo
    S = np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]])
    Q = np.eye(3) + np.sin(theta) * S + (1 - np.cos(theta)) * S @ S
    return R @ Q


def actitud(U: np.ndarray, t: np.ndarray) -> np.ndarray:
    R = np.zeros((len(t), 3, 3))
    R[0] = np.eye(3)
    for n in range(len(t) - 1):
        R[n + 1] = paso_actitud(R[n], U[n], U[n + 1], t[n + 1] - t[n])
    return R


def estabilidad(esquema: str, z: np.ndarray) -> np.ndarray:
    z = np.asarray(z)
    if esquema == "EE":
        return 1 + z
    if esquema == "EI":
        return 1 / (1 - z)
    if esquema == "CN":
        return (1 + z / 2) / (1 - z / 2)
    if esquema == "RK4":
        return 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24
    if esquema == "LF":
        return z + np.sqrt(1 + z**2)