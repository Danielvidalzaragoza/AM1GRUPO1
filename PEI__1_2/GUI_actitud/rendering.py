import numpy as np
from matplotlib.artist import Artist
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox

from .theme import TEXTO

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
