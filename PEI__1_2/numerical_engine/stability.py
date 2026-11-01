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

def obtener_todas_regiones_estabilidad(
    esquemas=("EE", "EI", "CN", "RK4", "LF"),
    resolucion=85
):    
    regiones = {}
    X, Y = None, None
    for esquema in esquemas:
        X, Y, rho = region_por_esquema(
            esquema, resolucion
        )
        regiones[esquema] = rho
    return X, Y, regiones

def region_por_esquema(esquema: str, resolucion: int = 85):
    from .integrators import paso_Cauchy
    x = np.linspace(-4.0, 2.0, resolucion)
    y = np.linspace(-3.0, 3.0, resolucion)
    X, Y = np.meshgrid(x, y)
    rho = np.full((resolucion, resolucion), np.nan)
    for j in range(resolucion):
        for i in range(resolucion):
            z = complex(x[i], y[j])
            # Ecuación referencia Amtonio
            F = lambda U, t: z * U
            J = lambda U, t: np.array([[z]], dtype=complex)
            uno = np.array([1.0 + 0j])
            cero = np.array([0.0 + 0j])
            try:
                with np.errstate(all="ignore"):
                    if esquema == "LF":
                        # Leapfrog necesita dos estados anteriores.
                        # Reconstruimos su matriz de amplificación.
                        a = paso_Cauchy(
                            "LF", uno, cero, 1.0, 0.0, F, J
                        )[0]
                        b = paso_Cauchy(
                            "LF", cero, uno, 1.0, 0.0, F, J
                        )[0]
                        M = np.array([
                            [a, b],
                            [1, 0]
                        ], dtype=complex)
                        raices = np.linalg.eigvals(M)
                        valor = np.max(np.abs(raices))
                    else:
                        resultado = paso_Cauchy(
                            esquema, uno, None,
                            1.0, 0.0, F, J
                        )
                        valor = abs(resultado[0])
                    if np.isfinite(valor):
                        rho[j, i] = valor

            except (RuntimeError, ValueError, ZeroDivisionError,
                    FloatingPointError, np.linalg.LinAlgError):
                # Puntos singulares o donde el solver falla.
                pass
    return X, Y, rho
