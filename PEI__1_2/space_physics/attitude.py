import numpy as np

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
