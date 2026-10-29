import numpy as np

from .dynamics import jacobiano

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
