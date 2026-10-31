import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def _alinear_con_z(v):
    v = v / np.linalg.norm(v)
    eje = np.cross(v, [0, 0, 1])
    s, c = np.linalg.norm(eje), v[2]
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = eje / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K

def _superficie_elipsoide(semiejes, limite, paralelos=16, meridianos=24):
    """Genera caras poligonales 3D para un elipsoide sólido transúcido."""
    a, b, c = semiejes
    u = np.linspace(0, np.pi, paralelos)
    v = np.linspace(0, 2 * np.pi, meridianos)
    U, V = np.meshgrid(u, v, indexing="ij")
    X = a * np.sin(U) * np.cos(V)
    Y = b * np.sin(U) * np.sin(V)
    Z = c * np.cos(U)
    
    fuera = (np.abs(X) > limite) | (np.abs(Y) > limite) | (np.abs(Z) > limite)
    verts = []
    for i in range(paralelos - 1):
        for j in range(meridianos - 1):
            if not (fuera[i, j] or fuera[i+1, j] or fuera[i+1, j+1] or fuera[i, j+1]):
                p0 = np.array([X[i, j], Y[i, j], Z[i, j]])
                p1 = np.array([X[i+1, j], Y[i+1, j], Z[i+1, j]])
                p2 = np.array([X[i+1, j+1], Y[i+1, j+1], Z[i+1, j+1]])
                p3 = np.array([X[i, j+1], Y[i, j+1], Z[i, j+1]])
                verts.append([p0, p1, p2, p3])
    return verts

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