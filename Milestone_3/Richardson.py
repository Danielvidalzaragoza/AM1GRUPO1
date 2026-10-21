import numpy as np

def error_richardson(
    coarse: np.ndarray,
    fine: np.ndarray,
    order: float,
) -> np.ndarray:
    """
    Estima el error de la aproximación fina mediante extrapolación de Richardson.
    `coarse` y `fine` deben representar la misma magnitud en los mismos tiempos,
    calculada con pasos h y h/n, respectivamente.
    """
    return (coarse - fine) / (2**order - 1)