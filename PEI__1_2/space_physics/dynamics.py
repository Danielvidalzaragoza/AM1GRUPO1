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
