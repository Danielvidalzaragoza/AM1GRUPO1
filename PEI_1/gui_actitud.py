"""GUI de actitud del PEI 1: un cuerpo rigido 3D girando libremente.

Capa de visualizacion. Reutiliza las otras dos capas:
  - space_physics: ecuaciones de Euler, Jacobiano, invariantes 2T y L^2,
    estabilidad espectral de los ejes, separatriz y actitud R(t).
  - numerical_engine: esquemas temporales, Richardson y |R(z)|.
y el catalogo de cuerpos (cuerpos.py) para la geometria y la inercia.

Para que la animacion sea fluida se usa "blitting": la parte estatica del
cuadro de mando se dibuja una vez y en cada fotograma solo se repintan los
elementos que cambian (cuerpo, ejes, curvas y textos de estado).
"""

import textwrap
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.artist import Artist
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.widgets import Button, RadioButtons, Slider
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

if __package__:
    from .cuerpos import CATALOGO
    from .numerical_engine import RK4, paso_Cauchy, richardson_error, stability_report, obtener_todas_regiones_estabilidad, datos_convergencia
    from .space_physics import (
        ecuaciones_rotacion,
        estabilidad_ejes,
        historial_invariantes,
        invariantes,
        jacobiano,
        paso_actitud,
        separatriz,
    )
else:
    from cuerpos import CATALOGO
    from numerical_engine import RK4, paso_Cauchy, richardson_error, stability_report, obtener_todas_regiones_estabilidad, datos_convergencia
    from space_physics import (
        ecuaciones_rotacion,
        estabilidad_ejes,
        historial_invariantes,
        invariantes,
        jacobiano,
        paso_actitud,
        separatriz,
    )


FONDO = "#0a0f1e"
TARJETA = "#111a2e"
BORDE = "#25314d"
CONTROL = "#1b2640"
CONTROL_HOVER = "#26345a"
TEXTO = "#e8ecf6"
TENUE = "#8b95ad"
ACENTO = "#4f8cff"
COLOR_EJES = ("#ff6b6b", "#ffd166", "#4cc9f0")
COLOR_L = "#c792ea"
COLOR_W = "#80ffb4"
COLOR_T = "#5b8def"
BIEN = "#80ffb4"
MAL = "#ff6b6b"
AVISO = "#ffb347"
MONO = ["Consolas", "DejaVu Sans Mono"]

ESQUEMAS = {
    "EE    Euler explícito": "EE",
    "EI    Euler implícito": "EI",
    "CN    Crank–Nicolson": "CN",
    "RK4  Runge–Kutta 4": "RK4",
    "LF    Leapfrog": "LF",
}
EJES = ("e₁   I₁ mínimo", "e₂   I₂ intermedio", "e₃   I₃ máximo")
SUB = "₁₂₃"
ANCHO, ALTO = 16.5, 8.4
TODAS = (0, 1, 2)
ESTADOS = {
    "listo": ("●  LISTO", TENUE),
    "marcha": ("●  EN CURSO", BIEN),
    "pausa": ("●  EN PAUSA", AVISO),
    "completado": ("●  COMPLETADO", ACENTO),
    "diverge": ("●  EL ESQUEMA DIVERGE", MAL),
}


def _aplicar_estilo():
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "text.color": TEXTO,
        "axes.labelcolor": TENUE,
        "axes.edgecolor": BORDE,
        "xtick.color": TENUE,
        "ytick.color": TENUE,
        "grid.color": BORDE,
        "grid.linewidth": 0.6,
        "figure.facecolor": FONDO,
        "axes.facecolor": TARJETA,
        "text.hinting": "no_hinting",
    })


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


class _Sprite(Artist):
    def __init__(self, fontsize, ha="center"):
        super().__init__()
        self._fuente = {"fontsize": fontsize, "fontweight": "bold"}
        self._ha = ha
        self._cache = {}
        self.texto, self.color, self.xy = "", TEXTO, (0.0, 0.0)

    def _imagen(self, dpi):
        clave = (self.texto, self.color, dpi)
        if clave not in self._cache:
            fig = Figure(figsize=(4, 1), dpi=dpi)
            FigureCanvasAgg(fig)
            fig.patch.set_alpha(0)
            fig.text(0.5, 0.5, self.texto, ha="center", va="center",
                     color=self.color, **self._fuente)
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            filas = np.flatnonzero(rgba[..., 3].any(axis=1))
            columnas = np.flatnonzero(rgba[..., 3].any(axis=0))
            self._cache[clave] = rgba[filas[0]:filas[-1] + 1,
                                      columnas[0]:columnas[-1] + 1][::-1].copy()
        return self._cache[clave]

    def caja(self, dpi):
        alto, ancho = self._imagen(dpi).shape[:2]
        x, y = self.xy
        x0 = x - ancho if self._ha == "right" else x - ancho / 2
        return Bbox.from_bounds(round(x0), round(y - alto / 2), ancho, alto)

    def draw(self, renderer):
        caja = self.caja(renderer.dpi)
        gc = renderer.new_gc()
        renderer.draw_image(gc, caja.x0, caja.y0, self._imagen(renderer.dpi))
        gc.restore()


class PanelActitud:
    def __init__(self):
        _aplicar_estilo()
        self.fig = plt.figure(figsize=(ANCHO, ALTO), facecolor=FONDO)
        self.cuerpo = CATALOGO[0]
        self.eje = 1
        self.esquema = "RK4"
        self.en_marcha = False
        self.estado = "listo"
        self._cargando = False
        self._fondo = None
        self._completo = True
        self._camara = None
        self._fotograma = 0
        self._caja_cuerpo = None

        self._crear_marco()
        self._crear_vista_cuerpo()
        self._crear_graficas()
        self._crear_controles()

        self.timer = self.fig.canvas.new_timer(interval=1)
        self.timer.add_callback(self._tick)
        conectar = self.fig.canvas.mpl_connect
        conectar("draw_event", self._al_dibujar)
        conectar("key_press_event", self._tecla)
        conectar("button_press_event", self._presionar)
        conectar("motion_notify_event", self._mover)
        conectar("button_release_event", self._soltar)
        conectar("close_event", lambda _: self.timer.stop())

        self._cargar_inercias()
        self.reiniciar()

    def _tarjeta(self, x, y, ancho, alto):
        self.fig.add_artist(FancyBboxPatch(
            (x, y), ancho, alto,
            boxstyle="round,pad=0,rounding_size=0.006",
            mutation_aspect=ANCHO / ALTO,
            transform=self.fig.transFigure,
            facecolor=TARJETA, edgecolor=BORDE, linewidth=0.8, zorder=-1,
        ))

    def _rotulo(self, x, y, texto):
        return self.fig.text(x, y, texto, fontsize=8, fontweight="bold",
                             color=TENUE)

    def _estilo_2d(self, ax):
        ax.set_facecolor(TARJETA)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            ax.spines[lado].set_color(BORDE)
        ax.tick_params(labelsize=7.5, colors=TENUE, length=2)
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.grid(True, color=BORDE, linewidth=0.5)

    def _crear_marco(self):
        self.fig.text(0.012, 0.968, "Dinámica de actitud", fontsize=17,
                      fontweight="bold", va="center")
        self.fig.text(
            0.012, 0.937,
            "Teorema del eje intermedio · efecto Dzhanibekov · PEI 1, "
            "Ampliación de Matemáticas 1",
            fontsize=9.5, color=TENUE, va="center",
        )
        self.ax_estado = self.fig.add_axes((0.74, 0.952, 0.147, 0.032))
        self.ax_estado.set_facecolor(FONDO)
        self.ax_estado.set(xticks=[], yticks=[])
        for lado in self.ax_estado.spines.values():
            lado.set_visible(False)
        self.texto_estado = self.ax_estado.text(1, 0.5, "", fontsize=11,
                                                fontweight="bold", ha="right",
                                                va="center")
        self.texto_reloj = self.fig.text(0.988, 0.968, "", fontsize=11,
                                         fontweight="bold", ha="right",
                                         va="center", animated=True)
        self._region_reloj = TransformedBbox(
            Bbox([[0.887, 0.95], [0.995, 0.986]]), self.fig.transFigure
        )
        self.fig.text(
            0.988, 0.937,
            "Arrastra una vista 3D para girar la cámara · Espacio: iniciar / pausar",
            fontsize=8.5, color=TENUE, ha="right", va="center",
        )
        for rect in (
            (0.008, 0.02, 0.165, 0.895),
            (0.178, 0.215, 0.380, 0.7),
            (0.178, 0.02, 0.380, 0.185),
            (0.570, 0.685, 0.422, 0.23),
            (0.570, 0.265, 0.204, 0.41),
            (0.786, 0.265, 0.206, 0.41),
            (0.570, 0.02, 0.204, 0.235),
            (0.786, 0.02, 0.206, 0.235),
        ):
            self._tarjeta(*rect)

    def _crear_vista_cuerpo(self):
        rect = (0.181, 0.218, 0.374, 0.694)
        self._region_cuerpo = TransformedBbox(Bbox.from_bounds(*rect),
                                              self.fig.transFigure)
        cielo = self.fig.add_axes(rect, zorder=0)
        cielo.set_axis_off()
        cielo.set(xlim=(0, 1), ylim=(0, 1))
        azar = np.random.default_rng(3)
        estrellas = 260
        brillo = np.column_stack((
            np.full((estrellas, 3), (0.85, 0.9, 1.0)),
            azar.uniform(0.15, 0.8, estrellas),
        ))
        cielo.scatter(azar.random(estrellas), azar.random(estrellas),
                      s=azar.uniform(0.3, 1.8, estrellas) ** 2, c=brillo,
                      linewidths=0)
        cielo.legend(
            [Line2D([], [], color=c, lw=2.4) for c in (*COLOR_EJES, COLOR_L, COLOR_W)],
            ["e₁  I mín", "e₂  I intermedio", "e₃  I máx", "L  momento angular",
             "ω  velocidad angular"],
            loc="lower left", ncol=5, frameon=False, fontsize=8,
            labelcolor=TEXTO, handlelength=1.3, columnspacing=1.1,
            borderaxespad=0.5,
        )

        ax = self.fig.add_axes(rect, projection="3d", zorder=1)
        ax.set_facecolor((0, 0, 0, 0))
        ax.set_axis_off()
        ax.set(xlim=(-1.35, 1.35), ylim=(-1.35, 1.35), zlim=(-1.35, 1.35))
        ax.set_box_aspect((1, 1, 1), zoom=1.45)
        ax.view_init(elev=16, azim=-58)
        ax.disable_mouse_rotation()
        self.ax_cuerpo = ax

        def capa(coleccion):
            coleccion.set_animated(True)
            ax.add_collection3d(coleccion)
            return coleccion

        s = np.linspace(0, 2 * np.pi, 90)
        circulo = np.column_stack((1.25 * np.cos(s), 1.25 * np.sin(s), 0 * s))
        self.referencias = capa(Line3DCollection(
            [circulo, [(0, 0, -1.35), (0, 0, 1.35)]],
            colors=[(1, 1, 1, 0.13), (1, 1, 1, 0.1)], linewidths=0.8,
            linestyles="--",
        ))
        vacio = [np.zeros((2, 3))]
        self.capa_detras = capa(Line3DCollection(vacio, capstyle="round"))
        self.poly = capa(Poly3DCollection(self.cuerpo.caras, linewidths=0))
        self.capa_delante = capa(Line3DCollection(vacio, capstyle="round"))

        self.rotulos_ejes = [_Sprite(11) for _ in COLOR_EJES]
        for k, rotulo in enumerate(self.rotulos_ejes):
            rotulo.texto, rotulo.color = f"e{SUB[k]}", COLOR_EJES[k]
        self.rotulo_L, self.rotulo_w = _Sprite(12), _Sprite(12)
        self.rotulo_L.texto, self.rotulo_L.color = "L", COLOR_L
        self.rotulo_w.texto, self.rotulo_w.color = "ω", COLOR_W

        self.texto_nombre = self.fig.text(0.188, 0.889, "", fontsize=13.5,
                                          fontweight="bold")
        self.texto_descripcion = self.fig.text(0.188, 0.874, "", fontsize=9,
                                               color=TENUE, va="top",
                                               linespacing=1.4)
        self.texto_aviso = self.fig.text(0.188, 0.818, "", fontsize=9,
                                         color=AVISO, va="top",
                                         linespacing=1.4)
        self.rotulo_volteos = _Sprite(12, ha="right")

    def _crear_graficas(self):
        self._rotulo(0.585, 0.893, "VELOCIDAD ANGULAR EN EJES CUERPO  ω(t)  [rad/s]")
        self.ax_w = self.fig.add_axes((0.605, 0.718, 0.368, 0.158))
        self._estilo_2d(self.ax_w)
        self.lineas_w = [
            self.ax_w.plot([], [], color=color, lw=1.4, label=f"ω{SUB[k]}",
                           animated=True)[0]
            for k, color in enumerate(COLOR_EJES)
        ]
        self.ax_w.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3,
                         fontsize=8, frameon=False, labelcolor=TEXTO,
                         handlelength=1.2, borderaxespad=0.1)

        self._rotulo(0.585, 0.653, "MAPA DE FASES")
        self._rotulo(0.801, 0.653, "ZONAS DE ESTABILIDAD")
        
        ax_f = self.fig.add_axes((0.577, 0.280, 0.190, 0.35), projection="3d",
                                 computed_zorder=False)
        ax_f.set_facecolor((0, 0, 0, 0))
        for k, (eje, color) in enumerate(zip((ax_f.xaxis, ax_f.yaxis, ax_f.zaxis),
                                             COLOR_EJES)):
            eje.set_pane_color((0.08, 0.12, 0.2, 0.6))
            eje.line.set_color(BORDE)
            eje.set_major_locator(MaxNLocator(3))
            eje.set_label_text(f"ω{SUB[k]}", color=color, fontsize=9)
            eje.labelpad = -8
        ax_f.tick_params(labelsize=6, colors=TENUE, pad=-2)
        ax_f.set_box_aspect((1, 1, 1), zoom=0.92)
        ax_f.view_init(elev=22, azim=-50)
        self.ax_fase = ax_f

        vacio = [np.zeros((2, 3))]
        self.malla_T = Line3DCollection(vacio, colors=[to_rgba(COLOR_T, 0.25)],
                                        linewidths=0.6, zorder=1)
        self.malla_L = Line3DCollection(vacio, colors=[to_rgba(COLOR_L, 0.25)],
                                        linewidths=0.6, zorder=1)
        self.separatriz = Line3DCollection(vacio,
                                           colors=[to_rgba("#f4f4f8", 0.75)],
                                           linewidths=1.1, linestyles="--",
                                           zorder=2)
        for coleccion in (self.malla_T, self.malla_L, self.separatriz):
            ax_f.add_collection3d(coleccion)
        (self.eq_estables,) = ax_f.plot([], [], [], "o", ms=4, color=BIEN,
                                      ls="none", zorder=3)
        (self.eq_inestables,) = ax_f.plot([], [], [], "X", ms=5.5, color=MAL,
                                        ls="none", zorder=3)
        (self.trayectoria,) = ax_f.plot([], [], [], color=COLOR_W, lw=1.3,
                                      zorder=4, animated=True)
        (self.punto,) = ax_f.plot([], [], [], "o", ms=5, color="white",
                                mec=COLOR_W, mew=1.5, zorder=5, animated=True)
        
        ax_f.legend(
            [Line2D([], [], color=COLOR_T), Line2D([], [], color=COLOR_L),
             Line2D([], [], color="#f4f4f8", ls="--"),
             Line2D([], [], color=BIEN, marker="o", ls="none", ms=4),
             Line2D([], [], color=MAL, marker="X", ls="none", ms=5)],
            ["2T", "L²", "separatriz", "centro", "silla"],
            loc="lower left", bbox_to_anchor=(0.04, -0.06), ncol=5, fontsize=6.5,
            frameon=False, labelcolor=TEXTO, handlelength=1.0, columnspacing=0.5,
        )

        ax_s = self.fig.add_axes((0.812, 0.30, 0.165, 0.33))
        ax_s.set_facecolor(TARJETA)
        for lado in ("top", "right"):
            ax_s.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            ax_s.spines[lado].set_color(BORDE)
        ax_s.tick_params(labelsize=6.5, colors=TENUE, length=2)
        ax_s.xaxis.set_major_locator(MaxNLocator(4))
        ax_s.yaxis.set_major_locator(MaxNLocator(4))
        ax_s.grid(True, color=BORDE, linewidth=0.5)
        ax_s.set_xlabel("Re(z)", fontsize=6.5, labelpad=1)
        ax_s.set_ylabel("Im(z)", fontsize=6.5, labelpad=1)
        self.ax_stab = ax_s

        
        
        self._rotulo(0.585, 0.232, "DERIVA DE INVARIANTES")
        self.ax_deriva = self.fig.add_axes(
            (0.605, 0.062, 0.165, 0.142)
        )
        self._estilo_2d(self.ax_deriva)
        self.ax_deriva.set_yscale("log")

        (self.linea_dT,) = self.ax_deriva.plot(
            [], [], color=COLOR_T, lw=1.5,
            label="|ΔT| / T₀", animated=True
        )
        (self.linea_dL,) = self.ax_deriva.plot(
            [], [], color=COLOR_L, lw=1.5,
            label="|ΔL²| / L₀²", animated=True
        )
        self.ax_deriva.legend(
            loc="lower right", bbox_to_anchor=(1.0, 1.0),
            ncol=2, fontsize=7, frameon=False,
            labelcolor=TEXTO, handlelength=1.0
        )
        self.ax_deriva.set_xlabel("t [s]", fontsize=8)

        
        self._rotulo(0.801, 0.232, "CONVERGENCIA DE INTEGRADORES")
        self.ax_convergencia = self.fig.add_axes(
            (0.812, 0.105, 0.165, 0.095)
        )
        self._estilo_2d(self.ax_convergencia)
        self.ax_convergencia.set_xlabel("log(N)", fontsize=8)
        self.ax_convergencia.set_ylabel("log(E)", fontsize=8)
        self.ax_convergencia.tick_params(labelsize=7)

        
        region_fase = TransformedBbox(Bbox.from_bounds(0.570, 0.265, 0.422, 0.41),
                                      self.fig.transFigure)
        self._grupos_graficas = (
            ([*self.lineas_w, self.texto_reloj], [self.ax_w.bbox, self._region_reloj]),
            ([self.trayectoria, self.punto], [region_fase]),
            ([self.linea_dT, self.linea_dL], [self.ax_deriva.bbox]),
        )

    def _radio(self, rect, etiquetas, activo, accion):
        ax = self.fig.add_axes(rect)
        ax.set_facecolor(TARJETA)
        for lado in ax.spines.values():
            lado.set_visible(False)
        n = len(etiquetas)
        radio = RadioButtons(
            ax, etiquetas, active=activo, activecolor=ACENTO,
            label_props={"color": [TEXTO] * n, "fontsize": [9.5] * n},
            radio_props={"edgecolor": [TENUE] * n, "s": [60] * n},
        )
        self._resaltar(radio)
        radio.on_clicked(accion)
        return radio

    @staticmethod
    def _resaltar(radio):
        for etiqueta in radio.labels:
            activa = etiqueta.get_text() == radio.value_selected
            etiqueta.set_color(TEXTO if activa else TENUE)
            etiqueta.set_fontweight("bold" if activa else "normal")

    def _boton(self, rect, texto, color, hover, accion):
        ax = self.fig.add_axes(rect)
        for lado in ax.spines.values():
            lado.set_visible(False)
        boton = Button(ax, texto, color=color, hovercolor=hover)
        boton.label.set(color=TEXTO, fontsize=9.5, fontweight="bold")
        boton.on_clicked(accion)
        return boton

    def _crear_controles(self):
        self._rotulo(0.015, 0.893, "CUERPO")
        self.radio_cuerpo = self._radio(
            (0.010, 0.705, 0.155, 0.18), [c.nombre for c in CATALOGO], 0,
            self.cambiar_cuerpo,
        )
        self._rotulo(0.015, 0.69, "EJE DE GIRO INICIAL")
        self.radio_eje = self._radio((0.010, 0.598, 0.155, 0.088), EJES,
                                     self.eje, self.cambiar_eje)
        self._rotulo(0.015, 0.583, "INTEGRADOR")
        self.radio_esquema = self._radio((0.010, 0.432, 0.155, 0.147),
                                         list(ESQUEMAS), 3, self.cambiar_esquema)

        self._rotulo(0.015, 0.412, "ANÁLISIS ESPECTRAL  (linealización)")
        mono = {"fontsize": 8.0, "family": MONO}
        self.filas_eje = []
        for k in range(3):
            y = 0.385 - 0.024 * k
            self.filas_eje.append((
                self.fig.text(0.012, y, "", color=COLOR_EJES[k],
                              fontweight="bold", **mono),
                self.fig.text(0.038, y, "", **mono),
                self.fig.text(0.168, y, "", ha="right", **mono),
            ))
        self.texto_invariantes = self.fig.text(0.015, 0.305, "", **mono)
        self.texto_cota = self.fig.text(0.015, 0.281, "", **mono)
        self.texto_esquema = self.fig.text(0.015, 0.257, "", **mono)
        self.texto_R = self.fig.text(0.015, 0.233, "", **mono)
        self.texto_richardson = self.fig.text(0.015, 0.201, "", **mono)
        self.texto_richardson_2 = self.fig.text(0.015, 0.177, "", **mono)

        self.boton_marcha = self._boton((0.012, 0.125, 0.075, 0.042), "",
                                        ACENTO, "#6aa0ff", self.alternar)
        self.boton_reinicio = self._boton((0.092, 0.125, 0.075, 0.042),
                                          "↺  Reiniciar", CONTROL,
                                          CONTROL_HOVER, self.reiniciar)
        self.boton_richardson = self._boton(
            (0.012, 0.065, 0.155, 0.045), "Estimar error (Richardson)",
            CONTROL, CONTROL_HOVER, self.calcular_richardson,
        )


        self._rotulo(0.190, 0.186, "INERCIA Y VELOCIDAD DE GIRO")
        self._rotulo(0.380, 0.186, "PERTURBACIÓN Y SIMULACIÓN")
        parametros = [
            ("I1", r"$I_1$", 0.05, 3.0, 1.0, "%.2f"),
            ("I2", r"$I_2$", 0.05, 3.0, 1.5, "%.2f"),
            ("I3", r"$I_3$", 0.05, 3.0, 2.0, "%.2f"),
            ("Omega", r"$\Omega$", 0.5, 6.0, 2.5, "%.2f rad/s"),
            ("eps", r"$\varepsilon$", 0.0, 0.2, 0.01, "%.3f"),
            ("dt", r"$\Delta t$", 0.005, 0.05, 0.01, "%.3f s"),
            ("tf", r"$t_f$", 5.0, 60.0, 30.0, "%.0f s"),
            ("vel", "vel.", 0.25, 4.0, 1.0, "×%.2f"),
        ]
        self.sliders, self._valores = {}, {}
        for n, (clave, etiqueta, minimo, maximo, inicial, formato) in enumerate(parametros):
            x, ancho = (0.210, 0.095) if n < 4 else (0.400, 0.085)
            y = 0.146 - 0.036 * (n % 4)
            ax = self.fig.add_axes((x, y, ancho, 0.022))
            ax.set_facecolor(TARJETA)
            slider = Slider(
                ax, etiqueta, minimo, maximo, valinit=inicial, valfmt=formato,
                color=ACENTO, track_color=CONTROL, initcolor="none",
                handle_style={"facecolor": "white", "edgecolor": ACENTO,
                              "size": 9},
            )
            slider.drawon = False
            slider.label.set(color=TEXTO, fontsize=9.5)
            slider.valtext.set_visible(False)
            caja = self.fig.add_axes((x + ancho + 0.003, y - 0.002, 0.042, 0.026))
            caja.set_facecolor(TARJETA)
            caja.set(xticks=[], yticks=[])
            for lado in caja.spines.values():
                lado.set_visible(False)
            valor = caja.text(0, 0.5, formato % inicial, va="center",
                              fontsize=8.0, color=TENUE)
            self._valores[clave] = (caja, valor, formato)
            
            if clave == "dt":
                slider.on_changed(lambda val, clave=clave: self._dt_cambiado(val))
            else:
                slider.on_changed(lambda val, clave=clave: self._slider_cambiado(clave, val))
                
            self.sliders[clave] = slider
        I1, I2, I3 = (self.sliders[k] for k in ("I1", "I2", "I3"))
        I1.slidermax, I2.slidermin, I2.slidermax, I3.slidermin = I2, I1, I3, I2

    def cambiar_cuerpo(self, nombre):
        self.cuerpo = next(c for c in CATALOGO if c.nombre == nombre)
        self._resaltar(self.radio_cuerpo)
        self._cargar_inercias()
        self.reiniciar()

    def cambiar_eje(self, etiqueta):
        self.eje = EJES.index(etiqueta)
        self._resaltar(self.radio_eje)
        self.reiniciar()

    def cambiar_esquema(self, etiqueta):
        self.esquema = ESQUEMAS[etiqueta]
        self._resaltar(self.radio_esquema)
        self._actualizar_estabilidad_plot()
        self.fig.canvas.draw_idle()
        self.reiniciar()

    def _dt_cambiado(self, valor):
        _, _, formato = self._valores["dt"]
        self._valores["dt"][1].set_text(formato % valor)
        if self._cargando:
            return
        self.At = valor
        self._actualizar_estabilidad_plot()
        self.fig.canvas.draw_idle()

    def _slider_cambiado(self, clave, valor):
        _, texto, formato = self._valores[clave]
        texto.set_text(formato % valor)
        if self._cargando:
            return
        if clave == "vel":
            self._completo = True
            self._refrescar()
        else:
            self.reiniciar()

    def _cargar_inercias(self):
        self._cargando = True
        try:
            for clave, valor in zip(("I1", "I2", "I3"),
                                    self.cuerpo.inercias_tipicas()):
                self.sliders[clave].set_val(valor)
        finally:
            self._cargando = False

    def _tecla(self, evento):
        if evento.key == " ":
            self.alternar()

    def _presionar(self, evento):
        if evento.inaxes is self.ax_cuerpo and evento.button == 1:
            self._camara = (evento.x, evento.y, self.ax_cuerpo.azim,
                            self.ax_cuerpo.elev)

    def _mover(self, evento):
        if self._camara is None or evento.x is None:
            return
        x0, y0, azimut, elevacion = self._camara
        caja = self.ax_cuerpo.bbox
        self.ax_cuerpo.view_init(
            elev=float(np.clip(elevacion - 180 * (evento.y - y0) / caja.height, -90, 90)),
            azim=azimut - 180 * (evento.x - x0) / caja.width,
        )
        self._dibujar_cuerpo()
        self._refrescar(grupos=())

    def _soltar(self, _):
        self._camara = None
        if self._completo:
            self._refrescar()

    def alternar(self, _=None):
        if self.en_marcha:
            self.en_marcha = False
            self.timer.stop()
            self.estado = "pausa"
        else:
            if self.n >= self.N or self.estado == "diverge":
                self.reiniciar()
            self.en_marcha = True
            self.estado = "marcha"
            self._reloj = time.perf_counter()
            self._acumulado = 0.0
            self.timer.start()
        self._actualizar_textos()
        self._refrescar()

    def reiniciar(self, _=None):
        s = self.sliders
        self.I = np.array([s["I1"].val, s["I2"].val, s["I3"].val])
        I1, I2, I3 = self.I
        self.Omega = s["Omega"].val
        self.W0 = np.full(3, s["eps"].val * self.Omega)
        self.W0[self.eje] = self.Omega
        self.tf = s["tf"].val
        self.N = int(np.ceil(self.tf / s["dt"].val))
        self.t = np.linspace(0, self.tf, self.N + 1)
        self.At = self.t[1] - self.t[0]
        self.F = lambda W, t: ecuaciones_rotacion(W, I1, I2, I3)
        self.J = lambda W, t: jacobiano(W, I1, I2, I3)

        self.U = np.zeros((self.N + 1, 3))
        self.U[0] = self.W0
        self.R = np.eye(3)
        self.n = 0
        self.energia0, self.momento0 = invariantes(self.W0, I1, I2, I3)
        self.P = _alinear_con_z(self.I * self.W0)
        self.puntas = np.zeros((self.N + 1, 3))
        self.puntas[0] = self.P[:, self.eje]
        self.volteos, self._lado = 0, 1
        self.caras, self.normales, self.realizable = self.cuerpo.deformar(I1, I2, I3)
        self.extension = np.abs(self.caras.reshape(-1, 3)).max(axis=0)

        self.estado = "marcha" if self.en_marcha else "listo"
        self._reloj = time.perf_counter()
        self._acumulado = 0.0

        self._preparar_fase()
        self._actualizar_analisis()
        self._actualizar_estabilidad_plot()
        self.mostrar_convergencia()
        self._completo = True
        self._dibujar()

    def _avanzar(self):
        n = self.n
        try:
            with np.errstate(all="ignore"):
                if self.esquema == "LF" and n == 0:
                    nuevo = RK4(self.U[0], self.At, self.t[0], self.F)
                else:
                    anterior = self.U[n - 1] if n > 0 else None
                    nuevo = paso_Cauchy(self.esquema, self.U[n], anterior,
                                        self.At, self.t[n], self.F, self.J)
        except (RuntimeError, np.linalg.LinAlgError):
            nuevo = None
        if (nuevo is None or not np.all(np.isfinite(nuevo))
                or np.linalg.norm(nuevo) > 1e3 * np.linalg.norm(self.W0)):
            self.en_marcha = False
            self.timer.stop()
            self.estado = "diverge"
            return False

        self.U[n + 1] = nuevo
        self.R = paso_actitud(self.R, self.U[n], nuevo, self.At)
        self.n = n + 1
        punta = self.P @ self.R[:, self.eje]
        self.puntas[self.n] = punta
        if punta[2] * self._lado < -0.5:
            self._lado = -self._lado
            self.volteos += 1
        return True

    def _tick(self):
        if not self.en_marcha:
            return
        ahora = time.perf_counter()
        self._acumulado += min(ahora - self._reloj, 0.25) * self.sliders["vel"].val
        self._reloj = ahora
        pasos = min(int(self._acumulado / self.At), self.N - self.n, 500)
        self._acumulado = min(self._acumulado - pasos * self.At, self.At)
        for _ in range(pasos):
            if not self._avanzar():
                break
        if self.n >= self.N and self.estado == "marcha":
            self.en_marcha = False
            self.timer.stop()
            self.estado = "completado"
        self._fotograma += 1
        self._dibujar(grupos=(self._fotograma % 3,) if self.en_marcha else TODAS)

    def calcular_richardson(self, _=None):
        if self.en_marcha:
            self.alternar()
        self.texto_richardson.set(text="Richardson: calculando…", color=TENUE)
        self.texto_richardson_2.set_text("")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        try:
            with np.errstate(all="ignore"):
                _, _, error, orden = richardson_error(
                    self.esquema, self.W0, self.N, 0, self.tf, self.F, self.J
                )
        except (RuntimeError, np.linalg.LinAlgError):
            orden = np.nan
            error = np.full((1, 3), np.nan)
        if np.isfinite(orden):
            self.texto_richardson.set(
                text=f"Richardson  orden p ≈ {orden:.2f}", color=TEXTO
            )
            self.texto_richardson_2.set(
                text=f"‖E(t_f)‖ ≈ {np.linalg.norm(error[-1]):.2e}", color=TEXTO
            )
        else:
            self.texto_richardson.set(text="Richardson: orden no definido",
                                      color=AVISO)
            self.texto_richardson_2.set(
                text="(solución estacionaria o esquema divergente)", color=TENUE
            )
        self._completo = True
        self._refrescar()
    def mostrar_convergencia(self, _=None):
        ax = self.ax_convergencia
        ax.clear()

        self._estilo_2d(ax)

        colores = {
            "EE": "#ff6b6b",
            "EI": "#80ffb4",
            "CN": "#4cc9f0",
            "RK4": "#ffd166",
            "LF": "#c792ea"
        }

        for esquema in ("EE", "EI", "CN", "RK4", "LF"):

            N, errores = datos_convergencia(
                esquema,
                self.W0,
                0,
                min(self.tf, 2.0),
                self.F,
                self.J
            )

            validos = (
                np.isfinite(errores)
                & (errores > 0)
                & np.isfinite(N)
                & (N > 0)
            )

            if np.count_nonzero(validos) >= 2:

                x = np.log(N[validos])
                y = np.log(errores[validos])
                n_ajuste = min(5, len(x))

                pendiente = np.polyfit(
                    x[:n_ajuste],
                    y[:n_ajuste],
                    1
                )[0]

                orden = -pendiente

                ax.plot(
                    x,
                    y,
                    "o-",
                    color=colores[esquema],
                    markersize=3,
                    linewidth=1.6,
                    label=f"{esquema} (p={orden:.2f})"
                )

        ax.set_xlabel("log(N)", fontsize=8, labelpad=1)
        ax.set_ylabel("log(E)", fontsize=8, labelpad=1)
        ax.tick_params(labelsize=7)

        ax.legend(
            fontsize=7,
            frameon=False,
            labelcolor=TEXTO,
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.32),
            handlelength=1.2,
            columnspacing=0.8
        )


        
    def _preparar_fase(self, limite=None):
        I1, I2, I3 = self.I
        semiejes_T = np.sqrt(2 * self.energia0 / self.I)
        semiejes_L = np.sqrt(self.momento0) / self.I
        if limite is None:
            limite = 1.3 * _cotas_polhodia(self.I, self.energia0, self.momento0).max()
        self._limite_fase = limite
        self.malla_T.set_segments(_malla_elipsoide(semiejes_T, limite))
        self.malla_L.set_segments(_malla_elipsoide(semiejes_L, limite))
        if I2 - I1 > 1e-6 and I3 - I2 > 1e-6:
            curva = separatriz(I1, I2, I3, self.energia0)
            self.separatriz.set_segments([curva, curva * [1, 1, -1]])
        else:
            self.separatriz.set_segments([np.zeros((2, 3))])
        a1, a2, a3 = np.where(semiejes_T <= limite, semiejes_T, np.nan)
        self.eq_estables.set_data_3d(np.array([[a1, -a1, 0, 0], [0, 0, 0, 0],
                                               [0, 0, a3, -a3]]))
        self.eq_inestables.set_data_3d(np.array([[0, 0], [a2, -a2], [0, 0]]))
        self.ax_fase.set(xlim=(-limite, limite), ylim=(-limite, limite),
                         zlim=(-limite, limite))

        self._limite_w = 1.15 * np.max(np.minimum(semiejes_T, semiejes_L))
        self.ax_w.set(xlim=(0, self.tf), ylim=(-self._limite_w, self._limite_w))
        self._techo_deriva = 1e-12
        self.ax_deriva.set(xlim=(0, self.tf), ylim=(1e-16, self._techo_deriva))

    def _actualizar_estabilidad_plot(self):
        self.ax_stab.clear()
        self.ax_stab.set_facecolor(TARJETA)
        for lado in ("top", "right"):
            self.ax_stab.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            self.ax_stab.spines[lado].set_color(BORDE)
        self.ax_stab.tick_params(labelsize=6.5, colors=TENUE, length=2)
        self.ax_stab.xaxis.set_major_locator(MaxNLocator(4))
        self.ax_stab.yaxis.set_major_locator(MaxNLocator(4))
        self.ax_stab.grid(True, color=BORDE, linewidth=0.5)

        X, Y, regiones = obtener_todas_regiones_estabilidad()
        colores_esq = {
            "EE": "#ff6b6b",
            "EI": "#80ffb4",
            "CN": "#4cc9f0",
            "RK4": "#ffd166",
            "LF": "#c792ea"
        }

        for esq, limite in regiones.items():
            color = colores_esq.get(esq, "#ffffff")
            self.ax_stab.contourf(X, Y, limite, levels=[0, 1.0], colors=[color], alpha=0.10)
            self.ax_stab.contour(X, Y, limite, levels=[1.0], colors=[color], linewidths=1.2)

        self.ax_stab.axhline(0, color=BORDE, linewidth=0.8)
        self.ax_stab.axvline(0, color=BORDE, linewidth=0.8)

        _, _, autovalores = estabilidad_ejes(self.I[0], self.I[1], self.I[2], self.Omega)
        lam_act = autovalores[f"I{self.eje + 1}"]
        z_act = self.At * lam_act
        self.ax_stab.scatter(z_act.real, z_act.imag, color=COLOR_EJES[self.eje], s=35, zorder=6, marker="o", edgecolor="black", linewidths=1)

        handles = [Line2D([0], [0], color=colores_esq[esq], lw=1.5, label=esq) for esq in ("EE", "EI", "CN", "RK4", "LF")]
        handles.append(Line2D([0], [0], color=COLOR_EJES[self.eje], marker="o", ls="none", ms=5, markeredgecolor="black", markeredgewidth=1, label="Autvalores eje activo"))
        
        self.ax_stab.legend(handles=handles, loc="lower left", fontsize=5.5, frameon=False, labelcolor=TEXTO, handlelength=1.0, borderaxespad=0.2, ncol=3)
        
        self.ax_stab.set_xlim(-3.2, 0.8)
        self.ax_stab.set_ylim(-1.8, 2.3)
        self.ax_stab.set_xlabel("Re(z)", fontsize=6.5, labelpad=1)
        self.ax_stab.set_ylabel("Im(z)", fontsize=6.5, labelpad=1)

    def _actualizar_analisis(self):
        I1, I2, I3 = self.I
        _, _, autovalores = estabilidad_ejes(I1, I2, I3, self.Omega)
        for k, (etiqueta, valor, veredicto) in enumerate(self.filas_eje):
            lam = autovalores[f"I{k + 1}"]
            lam = lam[np.argmax(np.abs(lam))]
            if abs(lam.real) > 1e-9:
                texto, juicio, color = f"λ = ±{abs(lam.real):.3f}", "silla · inestable", MAL
            elif abs(lam.imag) > 1e-9:
                texto, juicio, color = f"λ = ±{abs(lam.imag):.3f}i", "centro · estable", BIEN
            else:
                texto, juicio, color = "λ = 0", "degenerado", TENUE
            etiqueta.set_text(("▸" if k == self.eje else " ") + f"e{SUB[k]}")
            valor.set(text=texto, color=TEXTO if k == self.eje else TENUE)
            veredicto.set(text=juicio, color=color)

        dos_T = 2 * self.energia0
        self.texto_invariantes.set(
            text=f"2T = {dos_T:.4f}    L² = {self.momento0:.4f}", color=TEXTO
        )
        dentro = 2 * I1 * self.energia0 <= self.momento0 * (1 + 1e-12) and \
            self.momento0 <= 2 * I3 * self.energia0 * (1 + 1e-12)
        self.texto_cota.set(
            text="2I₁T ≤ L² ≤ 2I₃T  " + ("✓" if dentro else "✗"),
            color=BIEN if dentro else MAL,
        )

        lam = autovalores[f"I{self.eje + 1}"]
        _, maximo = stability_report(self.esquema, lam, self.At)
        if np.max(np.abs(lam.real)) > 1e-9:
            juicio, color = "crece (modo inestable)", AVISO
        elif abs(maximo - 1) < 1e-9:
            juicio, color = "conserva", BIEN
        elif maximo > 1:
            juicio, color = "amplifica", MAL
        else:
            juicio, color = "disipa", AVISO
        self.texto_esquema.set(
            text=f"{self.esquema} en e{SUB[self.eje]}:  |R(Δt·λ)|max", color=TEXTO
        )
        self.texto_R.set(text=f"  = {maximo:.6f}  → {juicio}", color=color)
        self.texto_richardson.set(text="Richardson: pulsa el botón inferior",
                                  color=TENUE)
        self.texto_richardson_2.set_text("")

        self.texto_nombre.set_text(self.cuerpo.nombre)
        self.texto_descripcion.set_text(textwrap.fill(self.cuerpo.descripcion, 40))
        self.texto_aviso.set_text("" if self.realizable else textwrap.fill(
            "⚠ I₃ > I₁ + I₂: ningún sólido real tiene estos momentos "
            "(se dibuja casi plano).", 44))

    def _dibujar_cuerpo(self):
        M = self.P @ self.R
        vertices = self.caras @ M.T
        self.poly.set_verts(vertices)
        self._vertices = vertices.reshape(-1, 3)

        azimut = np.radians(self.ax_cuerpo.azim)
        elevacion = np.radians(self.ax_cuerpo.elev)
        vista = np.array([np.cos(elevacion) * np.cos(azimut),
                          np.cos(elevacion) * np.sin(azimut),
                          np.sin(elevacion)])
        derecha = np.cross([0, 0, 1], vista)
        derecha = derecha / np.linalg.norm(derecha) if np.linalg.norm(derecha) > 1e-6 \
            else np.array([1.0, 0, 0])
        arriba = np.cross(vista, derecha)
        luz = 0.6 * vista + 0.55 * arriba - 0.45 * derecha
        luz /= np.linalg.norm(luz)
        medio = (luz + vista) / np.linalg.norm(luz + vista)
        normales = self.normales @ M.T
        difusa = np.abs(normales @ luz)
        brillo = np.abs(normales @ medio) ** 28
        base = self.cuerpo.colores
        rgb = np.clip(base[:, :3] * (0.32 + 0.68 * difusa)[:, None]
                      + 0.28 * brillo[:, None], 0, 1)
        self.poly.set_facecolor(np.column_stack((rgb, base[:, 3])))

        segmentos, colores, grosores = [], [], []

        def segmento(p0, p1, color, grosor):
            segmentos.append((p0, p1))
            colores.append(color)
            grosores.append(grosor)

        anclas = []
        for k in range(3):
            d, e = M[:, k], self.extension[k]
            segmento(d * (e + 0.04), d * (e + 0.3), to_rgba(COLOR_EJES[k]), 2.6)
            segmento(-d * (e + 0.04), -d * (e + 0.2), to_rgba(COLOR_EJES[k], 0.55), 1.4)
            anclas.append((self.rotulos_ejes[k], d * (e + 0.4)))

        w = self.U[self.n]
        L = M @ (self.I * w)
        L /= np.linalg.norm(L)
        w = M @ w
        w /= np.linalg.norm(w)
        segmento(1.04 * L, 1.3 * L, to_rgba(COLOR_L, 0.9), 3.2)
        segmento(1.04 * w, 1.2 * w, to_rgba(COLOR_W), 1.8)
        anclas += [(self.rotulo_L, 1.34 * L + 0.13 * derecha),
                   (self.rotulo_w, 1.2 * w - 0.13 * derecha)]
        self._anclas = anclas

        segmentos = np.array(segmentos)
        colores = np.array(colores)
        grosores = np.array(grosores)
        inicio = max(0, self.n - int(4 / self.At))
        paso = max(1, (self.n - inicio) // 80)
        puntos = self.puntas[inicio:self.n + 1:paso] * (self.extension[self.eje] + 0.3)
        if len(puntos) >= 2:
            estela = np.stack((puntos[:-1], puntos[1:]), axis=1)
            tonos = np.tile(to_rgba(COLOR_EJES[self.eje]), (len(estela), 1))
            tonos[:, 3] = np.linspace(0, 0.85, len(estela))
            segmentos = np.concatenate((estela, segmentos))
            colores = np.concatenate((tonos, colores))
            grosores = np.concatenate((np.full(len(estela), 1.6), grosores))

        delante = segmentos.mean(axis=1) @ vista >= 0
        for capa, elegidos in ((self.capa_delante, delante),
                               (self.capa_detras, ~delante)):
            if elegidos.any():
                capa.set_segments(segmentos[elegidos])
                capa.set_color(colores[elegidos])
                capa.set_linewidth(grosores[elegidos])
            else:
                capa.set_segments([np.zeros((2, 3))])
                capa.set_color([(0, 0, 0, 0)])
        self._segmentos = segmentos

    def _actualizar_graficas(self):
        n = self.n
        indices = np.r_[np.arange(0, n, max(1, n // 2500)), n]
        tiempos, estados = self.t[indices], self.U[indices]
        for k, linea in enumerate(self.lineas_w):
            linea.set_data(tiempos, estados[:, k])
        pico = np.abs(estados).max()
        if pico > self._limite_w:
            self._limite_w = 1.2 * pico
            self.ax_w.set_ylim(-self._limite_w, self._limite_w)
            self._completo = True

        energia, momento = historial_invariantes(estados, *self.I)
        deriva_T = np.maximum(np.abs(energia / self.energia0 - 1), 1e-16)
        deriva_L = np.maximum(np.abs(momento / self.momento0 - 1), 1e-16)
        self.linea_dT.set_data(tiempos, deriva_T)
        self.linea_dL.set_data(tiempos, deriva_L)
        peor = max(deriva_T.max(), deriva_L.max())
        if peor > self._techo_deriva:
            self._techo_deriva = 10.0 ** (np.ceil(np.log10(peor)) + 1)
            self.ax_deriva.set_ylim(1e-16, self._techo_deriva)
            self._completo = True

        self.trayectoria.set_data_3d(estados.T)
        self.punto.set_data_3d(estados[-1:].T)
        if np.abs(estados[-1]).max() > self._limite_fase:
            self._preparar_fase_ampliada(1.3 * np.abs(estados[-1]).max())

    def _preparar_fase_ampliada(self, limite):
        limite_w, techo = self._limite_w, self._techo_deriva
        self._preparar_fase(limite)
        self._limite_w, self._techo_deriva = limite_w, techo
        self.ax_w.set_ylim(-limite_w, limite_w)
        self.ax_deriva.set_ylim(1e-16, techo)
        self._completo = True

    def _actualizar_textos(self):
        nombre, color = ESTADOS[self.estado]
        boton = "❚❚  Pausar" if self.en_marcha else "▶  Iniciar"
        cambio = (self.texto_estado.get_text() != nombre
                  or self.boton_marcha.label.get_text() != boton)
        self.texto_estado.set(text=nombre, color=color)
        self.boton_marcha.label.set_text(boton)
        self.texto_reloj.set(text=f"t = {self.t[self.n]:.2f} / {self.tf:.0f} s",
                             color=color)
        self.rotulo_volteos.texto = f"Volteos de e{SUB[self.eje]}: {self.volteos}"
        self.rotulo_volteos.color = COLOR_EJES[self.eje]
        if cambio:
            self._actualizar_fondo(self.boton_marcha.ax, self.ax_estado)

    def _dibujar(self, grupos=TODAS):
        self._dibujar_cuerpo()
        self._actualizar_graficas()
        self._actualizar_textos()
        self._refrescar(grupos)

    def _pintar_cuerpo(self, renderer):
        ax = self.ax_cuerpo
        ax.M = ax.get_proj()
        ax.invM = np.linalg.inv(ax.M)
        for capa in (self.referencias, self.capa_detras, self.poly,
                     self.capa_delante):
            capa.do_3d_projection()
            capa.draw(renderer)

        anclas = np.array([punto for _, punto in self._anclas])
        puntos = np.concatenate((self._vertices, self._segmentos.reshape(-1, 3),
                                 anclas))
        x, y, _ = proj3d.proj_transform(*puntos.T, ax.M)
        pixeles = ax.transData.transform(np.column_stack((x, y)))
        margen = 8 * renderer.dpi / 100
        cajas = [Bbox([pixeles.min(axis=0) - margen, pixeles.max(axis=0) + margen])]
        for (rotulo, _), xy in zip(self._anclas, pixeles[-len(anclas):]):
            rotulo.xy = xy
            rotulo.draw(renderer)
            cajas.append(rotulo.caja(renderer.dpi))
        self.rotulo_volteos.xy = self.fig.transFigure.transform((0.550, 0.895))
        self.rotulo_volteos.draw(renderer)
        return Bbox.union(cajas)

    def _pintar_graficas(self, renderer, grupos=TODAS):
        regiones = []
        for indice in grupos:
            artistas, zonas = self._grupos_graficas[indice]
            for artista in artistas:
                artista.draw(renderer)
            regiones += zonas
        return regiones

    def _actualizar_fondo(self, *ejes):
        lienzo = self.fig.canvas
        if self._fondo is None or self._completo or not lienzo.supports_blit:
            return
        lienzo.restore_region(self._fondo)
        for ax in ejes:
            self.fig.draw_artist(ax)
        self._fondo = lienzo.copy_from_bbox(self.fig.bbox)
        for ax in ejes:
            lienzo.blit(ax.bbox)

    def _al_dibujar(self, _):
        lienzo = self.fig.canvas
        self._fondo = lienzo.copy_from_bbox(self.fig.bbox)
        renderer = lienzo.get_renderer()
        self._caja_cuerpo = self._pintar_cuerpo(renderer)
        self._pintar_graficas(renderer)
        self._volteos_en_pantalla = (self.rotulo_volteos.texto,
                                     self.rotulo_volteos.color)
        self._caja_volteos = self.rotulo_volteos.caja(renderer.dpi)

    def _refrescar(self, grupos=TODAS):
        lienzo = self.fig.canvas
        arrastrado = next((c for c, s in self.sliders.items() if s.drag_active),
                          None)
        if self._fondo is None or not lienzo.supports_blit or (
                self._completo and arrastrado is None):
            self._completo = False
            lienzo.draw_idle()
            return
        lienzo.restore_region(self._fondo)
        renderer = lienzo.get_renderer()
        regiones = []
        if arrastrado is not None:
            slider = self.sliders[arrastrado]
            caja_valor = self._valores[arrastrado][0]
            slider.label.set_visible(False)
            self.fig.draw_artist(slider.ax)
            slider.label.set_visible(True)
            self.fig.draw_artist(caja_valor)
            regiones += [slider.ax.bbox, caja_valor.bbox]

        caja = self._pintar_cuerpo(renderer)
        if self._camara is not None or self._caja_cuerpo is None:
            region = self._region_cuerpo
        else:
            region = Bbox.intersection(Bbox.union([caja, self._caja_cuerpo]),
                                       self._region_cuerpo)
        self._caja_cuerpo = caja
        if region is not None:
            regiones.append(region)
        volteos = (self.rotulo_volteos.texto, self.rotulo_volteos.color)
        if volteos != self._volteos_en_pantalla:
            nueva = self.rotulo_volteos.caja(renderer.dpi)
            regiones.append(Bbox.union([nueva, self._caja_volteos]))
            self._volteos_en_pantalla, self._caja_volteos = volteos, nueva

        regiones += self._pintar_graficas(renderer, grupos)
        for region in regiones:
            lienzo.blit(region)


def main():
    plt.rcParams["toolbar"] = "None"
    panel = PanelActitud()
    gestor = panel.fig.canvas.manager
    if gestor is not None:
        gestor.set_window_title("PEI 1 · Dinámica de actitud")
        ventana = getattr(gestor, "window", None)
        for maximizar in (lambda: ventana.state("zoomed"),
                          lambda: ventana.showMaximized()):
            try:
                maximizar()
                break
            except Exception:
                pass
    plt.show()


if __name__ == "__main__":
    main()