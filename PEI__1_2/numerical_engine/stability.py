import numpy as np

from space_physics import estabilidad

def stability_report(esquema: str, eigenvalues: np.ndarray, At: float):
    z = At * np.asarray(eigenvalues)
    if esquema == "LF":
        roots = np.stack((z + np.sqrt(1 + z**2), z - np.sqrt(1 + z**2)))
        maximum = np.max(np.abs(roots))
    else:
        maximum = np.max(np.abs(estabilidad(esquema, z)))
    return maximum <= 1, maximum

def obtener_region_estabilidad(esquema: str, resolucion: int = 300):
    x = np.linspace(-4.0, 2.0, resolucion)
    y = np.linspace(-3.0, 3.0, resolucion)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    R = estabilidad(esquema, Z)
    return X, Y, np.abs(R)

def obtener_todas_regiones_estabilidad(esquemas=("EE", "EI", "CN", "RK4", "LF"), resolucion=200):
    x = np.linspace(-4.0, 2.0, resolucion)
    y = np.linspace(-3.0, 3.0, resolucion)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    regiones = {}
    for esq in esquemas:
        R = estabilidad(esq, Z)
        regiones[esq] = np.abs(R)
    return X, Y, regiones
