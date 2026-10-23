"""Catalogo de cuerpos rigidos: geometria para dibujar y propiedades de masa.

Cada cuerpo es una union de piezas homogeneas (cajas, cilindros, troncos de
cono y elipsoides). Con ellas se calculan el centro de masas y el tensor de
inercia. Los cuerpos estan disenados para que los ejes x, y, z del cuerpo
sean los ejes principales de inercia minima, intermedia y maxima
(I1 < I2 < I3), como en el enunciado.
"""

import numpy as np


def _rgb(color):
    color = color.lstrip("#")
    return np.array([int(color[i:i + 2], 16) / 255 for i in (0, 2, 4)])


# Rotaciones que llevan el eje z local de una pieza al eje indicado del cuerpo.
_HACIA_EJE = {
    "x": np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float),
    "y": np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float),
    "z": np.eye(3),
}


def rotacion_z(angulo):
    c, s = np.cos(angulo), np.sin(angulo)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def normales(caras):
    """Normales unitarias de caras cuadrilateras (F, 4, 3)."""
    n = np.cross(caras[:, 2] - caras[:, 0], caras[:, 3] - caras[:, 1])
    modulo = np.linalg.norm(n, axis=1, keepdims=True)
    return np.divide(n, modulo, out=np.zeros_like(n), where=modulo > 0)


class Pieza:
    """Pieza homogenea con sus caras ya colocadas en ejes del cuerpo.

    `momentos` son los segundos momentos por unidad de masa respecto al
    centro de la pieza, (int x^2, int y^2, int z^2) / m, en ejes locales.
    """

    def __init__(self, caras, volumen, momentos, centro=(0, 0, 0),
                 rotacion=None, color="#cccccc", densidad=1.0, alpha=1.0):
        rotacion = np.eye(3) if rotacion is None else np.asarray(rotacion)
        self.centro = np.asarray(centro, dtype=float)
        self.caras = caras @ rotacion.T + self.centro
        self.masa = densidad * volumen
        self.covarianza = rotacion @ np.diag(momentos) @ rotacion.T
        rgb = np.atleast_2d(_rgb(color) if isinstance(color, str) else color)
        rgba = np.column_stack((rgb, np.full(len(rgb), alpha)))
        self.colores = np.broadcast_to(rgba, (len(caras), 4)).copy()


def caja(dimensiones, centro=(0, 0, 0), rotacion=None, **estilo):
    a, b, c = dimensiones
    x, y, z = a / 2, b / 2, c / 2
    vertices = np.array([
        [-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
        [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z],
    ])
    caras = vertices[[[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
                      [2, 3, 7, 6], [1, 2, 6, 5], [0, 4, 7, 3]]]
    momentos = (a**2 / 12, b**2 / 12, c**2 / 12)
    return Pieza(caras, a * b * c, momentos, centro, rotacion, **estilo)


def cilindro(radio, altura, eje="z", centro=(0, 0, 0), radio_final=None,
             lados=18, **estilo):
    """Prisma de `lados` caras a lo largo de `eje`.

    Con `radio_final` distinto de `radio` es un tronco de cono; su masa se
    aproxima por la del cilindro del mismo volumen.
    """
    r0 = radio
    r1 = radio if radio_final is None else radio_final
    phi = np.linspace(0, 2 * np.pi, lados + 1)
    aro = np.column_stack((np.cos(phi), np.sin(phi), np.zeros_like(phi)))
    abajo = aro * [r0, r0, 1] + [0, 0, -altura / 2]
    arriba = aro * [r1, r1, 1] + [0, 0, altura / 2]
    centro_abajo = np.broadcast_to([0, 0, -altura / 2], (lados, 3))
    centro_arriba = np.broadcast_to([0, 0, altura / 2], (lados, 3))
    caras = np.concatenate((
        np.stack((abajo[:-1], abajo[1:], arriba[1:], arriba[:-1]), axis=1),
        np.stack((centro_abajo, abajo[1:], abajo[:-1], abajo[:-1]), axis=1),
        np.stack((centro_arriba, arriba[:-1], arriba[1:], arriba[1:]), axis=1),
    ))
    area = lados / 2 * np.sin(2 * np.pi / lados) * (r0**2 + r0 * r1 + r1**2) / 3
    radio2 = area / np.pi
    momentos = (radio2 / 4, radio2 / 4, altura**2 / 12)
    return Pieza(caras, area * altura, momentos, centro, _HACIA_EJE[eje],
                 **estilo)


def elipsoide(semiejes, centro=(0, 0, 0), divisiones=(12, 20), relieve=None,
              **estilo):
    """Elipsoide de semiejes (a, b, c); `relieve(d)` deforma radialmente."""
    a, b, c = semiejes
    u = np.linspace(0, np.pi, divisiones[0] + 1)
    v = np.linspace(0, 2 * np.pi, divisiones[1] + 1)
    U, V = np.meshgrid(u, v, indexing="ij")
    d = np.stack(
        (np.sin(U) * np.cos(V), np.sin(U) * np.sin(V), np.cos(U)), axis=-1
    )
    factor = 1 if relieve is None else relieve(d)[..., None]
    P = d * factor * [a, b, c]
    caras = np.stack(
        (P[:-1, :-1], P[1:, :-1], P[1:, 1:], P[:-1, 1:]), axis=2
    ).reshape(-1, 4, 3)
    momentos = (a**2 / 5, b**2 / 5, c**2 / 5)
    return Pieza(caras, 4 / 3 * np.pi * a * b * c, momentos, centro,
                 **estilo)


def aro_eliptico(A, B, ancho, grosor, centro=(0, 0, 0), segmentos=30,
                 **estilo):
    """Aro eliptico en el plano xy formado por cajas tangentes."""
    phi = np.linspace(0, 2 * np.pi, segmentos + 1)
    puntos = np.column_stack((A * np.cos(phi), B * np.sin(phi)))
    piezas = []
    for p0, p1 in zip(puntos[:-1], puntos[1:]):
        d = p1 - p0
        medio = (p0 + p1) / 2
        piezas.append(caja(
            (np.linalg.norm(d) * 1.04, ancho, grosor),
            centro=(centro[0] + medio[0], centro[1] + medio[1], centro[2]),
            rotacion=rotacion_z(np.arctan2(d[1], d[0])),
            **estilo,
        ))
    return piezas


class Cuerpo:
    """Cuerpo rigido: caras normalizadas (radio envolvente 1) e inercia."""

    def __init__(self, nombre, descripcion, piezas):
        self.nombre = nombre
        self.descripcion = descripcion
        masa = sum(p.masa for p in piezas)
        centro = sum(p.masa * p.centro for p in piezas) / masa
        segundo_momento = sum(
            p.masa * (p.covarianza + np.outer(p.centro - centro, p.centro - centro))
            for p in piezas
        ) / masa
        caras = np.concatenate([p.caras for p in piezas]) - centro
        escala = 1 / np.linalg.norm(caras.reshape(-1, 3), axis=1).max()
        self.caras = caras * escala
        self.colores = np.concatenate([p.colores for p in piezas])
        segundo_momento = segundo_momento * escala**2
        # Tensor de inercia por unidad de masa: I = tr(C) Id - C.
        self.inercia = np.trace(segundo_momento) * np.eye(3) - segundo_momento
        self.segundos_momentos = np.diag(segundo_momento).copy()

    @property
    def momentos(self):
        return np.diag(self.inercia).copy()

    def inercias_tipicas(self, maximo=2.0):
        """(I1, I2, I3) del cuerpo, escalados para que I3 = maximo."""
        momentos = self.momentos / self.momentos.max() * maximo
        return tuple(float(I) for I in momentos)

    def deformar(self, I1, I2, I3):
        """Estira el cuerpo segun cada eje para que tenga inercias (I1, I2, I3).

        Los segundos momentos son Jx = (I2 + I3 - I1)/2, etc. Si alguno es
        negativo, los momentos violan la desigualdad triangular I3 <= I1 + I2
        y no existe ningun solido real con ellos (se dibuja casi plano).
        """
        J = 0.5 * np.array([I2 + I3 - I1, I1 + I3 - I2, I1 + I2 - I3])
        # Tolerancia para cuerpos planos (I1 + I2 = I3, como la raqueta).
        realizable = bool(J.min() > -1e-3 * J.max())
        J = np.maximum(J, 1e-3 * J.max())
        caras = self.caras * np.sqrt(J / self.segundos_momentos)
        caras /= np.linalg.norm(caras.reshape(-1, 3), axis=1).max()
        return caras, normales(caras), realizable


def _raqueta():
    marco = {"color": "#e9edf5"}
    piezas = aro_eliptico(0.5, 0.38, 0.055, 0.075, centro=(0.55, 0, 0), **marco)
    # Cordaje: cuerdas sin masa dentro del aro (semiejes interiores A, B).
    A, B = 0.47, 0.35
    cuerda = {"color": "#d9f99d", "densidad": 0.0}
    for y in np.linspace(-B, B, 10)[1:-1]:
        largo = 2 * A * np.sqrt(1 - (y / B) ** 2)
        piezas.append(caja((largo, 0.01, 0.01), centro=(0.55, y, 0), **cuerda))
    for x in np.linspace(-A, A, 12)[1:-1]:
        largo = 2 * B * np.sqrt(1 - (x / A) ** 2)
        piezas.append(caja((0.01, largo, 0.01), centro=(0.55 + x, 0, 0), **cuerda))
    angulo = np.radians(145)
    for signo in (1, -1):
        p0 = np.array([-0.08, 0.03 * signo])
        p1 = np.array([0.55 + 0.5 * np.cos(angulo), 0.38 * np.sin(angulo) * signo])
        d = p1 - p0
        medio = (p0 + p1) / 2
        piezas.append(caja((np.linalg.norm(d), 0.045, 0.06),
                           centro=(medio[0], medio[1], 0),
                           rotacion=rotacion_z(np.arctan2(d[1], d[0])), **marco))
    piezas += [
        caja((0.62, 0.075, 0.06), centro=(-0.38, 0, 0), **marco),
        caja((0.42, 0.095, 0.08), centro=(-0.86, 0, 0), color="#e63946"),
        caja((0.04, 0.11, 0.095), centro=(-1.085, 0, 0), color="#3d4566"),
    ]
    return Cuerpo(
        "Raqueta de tenis",
        "Eje intermedio: en el plano del cordaje, perpendicular al mango.",
        piezas,
    )


def _mango_en_t():
    acero = "#b8c2d6"
    piezas = [
        cilindro(0.08, 1.8, eje="x", color=acero),
        elipsoide((0.1, 0.12, 0.12), centro=(0.93, 0, 0), divisiones=(8, 14),
                  color="#8d99ae"),
        elipsoide((0.1, 0.12, 0.12), centro=(-0.93, 0, 0), divisiones=(8, 14),
                  color="#8d99ae"),
        cilindro(0.085, 0.95, eje="y", centro=(0, 0.555, 0), color="#d9a441"),
        cilindro(0.18, 0.17, eje="y", centro=(0, 1.0, 0), lados=6,
                 color="#9aa5b8"),
    ]
    return Cuerpo(
        "Mango en T (Dzhanibekov)",
        "Salyut 7, 1985: girando sobre el vástago, voltea cada pocos segundos.",
        piezas,
    )


def _satelite():
    piezas = [caja((0.42, 0.62, 0.36), color="#d4a24c")]
    for signo in (1, -1):
        piezas.append(caja((0.16, 0.035, 0.035), centro=(signo * 0.29, 0, 0),
                           color="#9aa5b8"))
        for i in range(4):
            for j in (-0.5, 0.5):
                piezas.append(caja(
                    (0.25, 0.2, 0.02),
                    centro=(signo * (0.5 + 0.265 * i), 0.215 * j, 0),
                    color="#2f5bd3", densidad=2.0,
                ))
    piezas += [
        cilindro(0.025, 0.12, eje="z", centro=(0, 0, 0.24), lados=10,
                 color="#9aa5b8"),
        elipsoide((0.2, 0.2, 0.045), centro=(0, 0, 0.33), divisiones=(8, 18),
                  color="#eef1f6"),
        cilindro(0.07, 0.1, eje="z", centro=(0, 0, -0.23), radio_final=0.04,
                 lados=14, color="#5a6275"),
    ]
    return Cuerpo(
        "Satélite con paneles",
        "Bus con paneles solares desplegados: I₂ e I₃ muy próximos.",
        piezas,
    )


def _libro():
    tapa = "#d1495b"
    piezas = [
        caja((1.3, 0.95, 0.03), centro=(0, 0, 0.115), color=tapa),
        caja((1.3, 0.95, 0.03), centro=(0, 0, -0.115), color=tapa),
        caja((1.3, 0.04, 0.26), centro=(0, -0.475, 0), color="#a83246"),
        caja((1.26, 0.92, 0.2), centro=(0, 0.005, 0), color="#f3ead3",
             densidad=0.8),
        caja((0.8, 0.07, 0.004), centro=(0, 0.15, 0.132), color="#f2c14e",
             densidad=0.0),
    ]
    return Cuerpo(
        "Libro",
        "Lánzalo girando sobre su eje intermedio: siempre da media vuelta.",
        piezas,
    )


def _asteroide():
    azar = np.random.default_rng(7)
    crateres = azar.normal(size=(10, 3))
    crateres /= np.linalg.norm(crateres, axis=1, keepdims=True)
    radios = azar.uniform(0.22, 0.45, len(crateres))

    def profundidad(d):
        hondo = np.zeros(d.shape[:-1])
        for c, r in zip(crateres, radios):
            distancia = np.linalg.norm(d - c, axis=-1) / r
            hondo += np.exp(-4 * distancia**2) - 0.45 * np.exp(-12 * (distancia - 1)**2)
        return hondo

    def relieve(d):
        ondas = np.sin(3 * d[..., 0] + 1) * np.sin(4 * d[..., 1] + 2) * np.sin(2 * d[..., 2] + 0.5)
        return 1 + 0.05 * ondas - 0.07 * profundidad(d)

    roca = elipsoide((1.0, 0.68, 0.46), divisiones=(16, 28), relieve=relieve,
                     color="#8b7d6b")
    direccion = roca.caras.mean(axis=1)
    direccion /= np.linalg.norm(direccion, axis=1, keepdims=True)
    tono = 0.88 + 0.22 * azar.random(len(direccion)) - 0.3 * np.clip(profundidad(direccion), 0, 1)
    roca.colores[:, :3] = np.clip(roca.colores[:, :3] * tono[:, None], 0, 1)
    return Cuerpo(
        "Asteroide triaxial",
        "Roca de semiejes a > b > c en rotación libre (tumbling).",
        [roca],
    )


def _cohete():
    blanco, rojo, tobera = "#e9edf5", "#e63946", "#5a6275"
    piezas = [
        cilindro(0.2, 1.5, eje="x", color=blanco),
        cilindro(0.2, 0.36, eje="x", centro=(0.93, 0, 0), radio_final=0.03,
                 color=rojo),
        cilindro(0.17, 0.22, eje="x", centro=(-0.86, 0, 0), radio_final=0.1,
                 color=tobera),
        cilindro(0.205, 0.07, eje="x", centro=(0.35, 0, 0), color="#2b2d42"),
    ]
    for signo in (1, -1):
        y = signo * 0.31
        piezas += [
            cilindro(0.1, 1.05, eje="x", centro=(-0.2, y, 0), lados=14,
                     color="#f2f4f8"),
            cilindro(0.1, 0.2, eje="x", centro=(0.425, y, 0), radio_final=0.02,
                     lados=14, color=rojo),
            cilindro(0.085, 0.12, eje="x", centro=(-0.785, y, 0),
                     radio_final=0.06, lados=14, color=tobera),
        ]
    return Cuerpo(
        "Cohete con boosters",
        "Etapa abandonada (basura espacial): su volteo es clave en ADR.",
        piezas,
    )


CATALOGO = [_raqueta(), _mango_en_t(), _satelite(), _libro(), _asteroide(), _cohete()]
