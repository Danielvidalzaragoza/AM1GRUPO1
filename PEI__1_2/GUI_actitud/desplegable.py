"""Menú desplegable para matplotlib (matplotlib.widgets no incluye uno)."""

import numpy as np
from matplotlib.patches import Rectangle

from .theme import ACENTO, BORDE, CONTROL, CONTROL_HOVER, TENUE, TEXTO


class Desplegable:
    """Cabecera con la opción activa y una lista que se abre hacia abajo.

    Solo mantiene su propio estado visual: el panel decide qué hacer al
    abrirse, cerrarse o elegir (``accion`` recibe la etiqueta elegida).
    """

    def __init__(self, fig, rect, opciones, activo, accion, zorder=30):
        self.fig = fig
        self.opciones = list(opciones)
        self.indice = activo
        self.accion = accion
        self.abierto = False
        self.resaltada = None
        x, y, ancho, alto = rect
        n = len(self.opciones)

        self.ax = fig.add_axes(rect, zorder=zorder)
        self._base(self.ax)
        self.texto = self.ax.text(0.05, 0.5, self.opciones[activo],
                                  transform=self.ax.transAxes, va="center",
                                  fontsize=9.5, fontweight="bold", color=TEXTO)
        self.flecha = self.ax.text(0.95, 0.5, "▾", transform=self.ax.transAxes,
                                   ha="right", va="center", fontsize=10,
                                   color=TENUE)

        self.ax_lista = fig.add_axes((x, y - n * alto, ancho, n * alto),
                                     zorder=zorder + 1)
        self._base(self.ax_lista)
        self.ax_lista.set(xlim=(0, 1), ylim=(n, 0))
        self.filas, self.etiquetas, self.marcas = [], [], []
        for k, opcion in enumerate(self.opciones):
            self.filas.append(self.ax_lista.add_patch(
                Rectangle((0, k), 1, 1, facecolor=CONTROL, edgecolor="none")))
            self.etiquetas.append(self.ax_lista.text(
                0.05, k + 0.5, opcion, va="center", fontsize=9.5, color=TENUE))
            self.marcas.append(self.ax_lista.text(
                0.95, k + 0.5, "●", ha="right", va="center", fontsize=7,
                color=ACENTO))
        self.ax_lista.set_visible(False)
        self._pintar()

    @staticmethod
    def _base(ax):
        ax.set_facecolor(CONTROL)
        ax.set(xticks=[], yticks=[])
        for lado in ax.spines.values():
            lado.set_color(BORDE)
            lado.set_linewidth(0.9)

    def _pintar(self):
        for k, (fila, etiqueta, marca) in enumerate(
                zip(self.filas, self.etiquetas, self.marcas)):
            activa, bajo_cursor = k == self.indice, k == self.resaltada
            fila.set_facecolor(CONTROL_HOVER if bajo_cursor else CONTROL)
            etiqueta.set(color=TEXTO if activa or bajo_cursor else TENUE,
                         fontweight="bold" if activa else "normal")
            marca.set_visible(activa)

    def abrir(self):
        self.abierto, self.resaltada = True, None
        self.ax_lista.set_visible(True)
        self.flecha.set_text("▴")
        self._pintar()

    def cerrar(self):
        self.abierto, self.resaltada = False, None
        self.ax_lista.set_visible(False)
        self.flecha.set_text("▾")

    def fila_en(self, evento):
        """Índice de la fila bajo el cursor, o None si está fuera de la lista."""
        if (not self.abierto or evento.inaxes is not self.ax_lista
                or evento.ydata is None):
            return None
        return int(np.clip(np.floor(evento.ydata), 0, len(self.opciones) - 1))

    def resaltar(self, fila):
        """Marca la fila bajo el cursor; devuelve True si hubo cambio."""
        if fila == self.resaltada:
            return False
        self.resaltada = fila
        self._pintar()
        return True

    def fijar(self, etiqueta):
        """Cambia la opción activa sin invocar la acción."""
        self.indice = self.opciones.index(etiqueta)
        self.texto.set_text(etiqueta)
        self._pintar()

    def elegir(self, fila):
        self.fijar(self.opciones[fila])
        self.cerrar()
        self.accion(self.opciones[fila])
