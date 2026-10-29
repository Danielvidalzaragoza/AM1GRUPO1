import numpy as np

def invariantes(W: np.ndarray, I1: float, I2: float, I3: float):
    energia = 0.5 * (I1 * W[0] ** 2 + I2 * W[1] ** 2 + I3 * W[2] ** 2)
    momento = (I1 * W[0]) ** 2 + (I2 * W[1]) ** 2 + (I3 * W[2]) ** 2
    return energia, momento

def historial_invariantes(U: np.ndarray, I1: float, I2: float, I3: float):
    energia = 0.5 * (I1 * U[:, 0] ** 2 + I2 * U[:, 1] ** 2 + I3 * U[:, 2] ** 2)
    momento = (I1 * U[:, 0]) ** 2 + (I2 * U[:, 1]) ** 2 + (I3 * U[:, 2]) ** 2
    return energia, momento
