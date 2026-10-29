import numpy as np

def _alinear_con_z(v):
    v = v / np.linalg.norm(v)
    eje = np.cross(v, [0, 0, 1])
    s, c = np.linalg.norm(eje), v[2]
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = eje / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K

def _malla_elipsoide(semiejes, limite, paralelos=7, meridianos=12, puntos=90):
    a, b, c = semiejes
    s = np.linspace(0, 2 * np.pi, puntos)
    lineas = []
    for theta in np.linspace(0, np.pi, paralelos + 2)[1:-1]:
        lineas.append(np.column_stack((
            a * np.sin(theta) * np.cos(s),
            b * np.sin(theta) * np.sin(s),
            np.full_like(s, c * np.cos(theta)),
        )))
    for phi in np.linspace(0, np.pi, meridianos, endpoint=False):
        lineas.append(np.column_stack((
            a * np.sin(s) * np.cos(phi),
            b * np.sin(s) * np.sin(phi),
            c * np.cos(s),
        )))
    for linea in lineas:
        linea[np.any(np.abs(linea) > limite, axis=1)] = np.nan
    return lineas

def _cotas_polhodia(I, energia, momento):
    I1, I2, I3 = I
    dos_T = 2 * energia
    D3 = max(dos_T * I3 - momento, 0.0)
    D1 = max(momento - dos_T * I1, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cotas = np.array([
            D3 / (I1 * (I3 - I1)),
            min(D3 / (I2 * (I3 - I2)), D1 / (I2 * (I2 - I1))),
            D1 / (I3 * (I3 - I1)),
        ])
    cotas = np.where(np.isfinite(cotas), cotas, np.inf)
    return np.sqrt(np.minimum(cotas, dos_T / np.asarray(I)))
