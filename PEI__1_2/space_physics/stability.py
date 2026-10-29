import numpy as np

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
