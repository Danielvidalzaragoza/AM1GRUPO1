import numpy as np


def error_richardson(
    coarse: np.ndarray,
    fine: np.ndarray,
    order: int,
) -> np.ndarray:
    """
    Estima el error de la aproximación fina mediante extrapolación de Richardson.

    `coarse` y `fine` deben representar la misma magnitud en los mismos tiempos,
    calculada con pasos h y h/2, respectivamente.
    """
    if order < 1:
        raise ValueError("El orden del método debe ser un entero positivo.")
    if coarse.shape != fine.shape:
        raise ValueError("Las aproximaciones coarse y fine deben tener la misma forma.")

    return (coarse - fine) / (2**order - 1)
