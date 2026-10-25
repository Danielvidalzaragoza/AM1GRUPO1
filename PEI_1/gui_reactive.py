"""GUI sencilla para simular la rotacion libre de un solido rigido."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons, Slider

if __package__:
    from .numerical_engine import paso_Cauchy, RK4, stability_report
    from .space_physics import (
        ecuaciones_rotacion,
        estabilidad_ejes,
        historial_invariantes,
        jacobiano,
        paso_actitud,
        separatriz,
    )
else:
    from numerical_engine import paso_Cauchy, RK4, stability_report
    from space_physics import (
        ecuaciones_rotacion,
        estabilidad_ejes,
        historial_invariantes,
        jacobiano,
        paso_actitud,
        separatriz,
    )


class RotationDashboard:
    def __init__(self):
        self.esquema = "RK4"
        self.en_marcha = False
        self.inicializada = False
        self.reiniciar_al_iniciar = False
        self.fig = plt.figure(figsize=(13, 9))
        self.fig.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.39)

        self.ax_omega = self.fig.add_subplot(2, 2, 1)
        self.ax_error = self.fig.add_subplot(2, 2, 2)
        self.ax_fase = self.fig.add_subplot(2, 2, 3, projection="3d")
        self.ax_actitud = self.fig.add_subplot(2, 2, 4, projection="3d")

        parametros = [
            ("I1", 0.2, 1.0, 0.8),
            ("I2", 1.1, 2.0, 1.5),
            ("I3", 2.1, 4.0, 3.0),
            ("w1(0)", -1.0, 1.0, 0),
            ("w2(0)", -2.0, 2.0, 0),
            ("w3(0)", -1.0, 1.0, 0),
            ("dt", 0.002, 0.08, 0.02),
            ("tf", 1.0, 20.0, 8.0),
        ]
        self.sliders = {}
        for n, (nombre, minimo, maximo, inicial) in enumerate(parametros):
            x = 0.10 if n < 4 else 0.49
            y = 0.32 - (n % 4) * 0.065
            ax = self.fig.add_axes([x, y, 0.28, 0.025])
            self.sliders[nombre] = Slider(
                ax, nombre, minimo, maximo, valinit=inicial
            )
            self.sliders[nombre].on_changed(self.cambiar_parametros)

        ax_esquema = self.fig.add_axes([0.82, 0.12, 0.12, 0.23])
        self.selector = RadioButtons(
            ax_esquema, ("EE", "EI", "CN", "RK4", "LF"), active=3
        )
        self.selector.on_clicked(self.cambiar_esquema)

        ax_simulacion = self.fig.add_axes([0.78, 0.04, 0.18, 0.045])
        self.boton_simulacion = Button(ax_simulacion, "Iniciar simulacion")
        self.boton_simulacion.on_clicked(self.iniciar_pausar)
        self.estado = self.fig.text(0.08, 0.025, "Configura los parametros y pulsa Iniciar.", fontsize=9)

        self.timer = self.fig.canvas.new_timer(interval=10)
        self.timer.add_callback(self.avanzar_paso)

    def cambiar_esquema(self, esquema):
        self.esquema = esquema
        self.reiniciar()

    def cambiar_parametros(self, _=None):
        self.reiniciar()

    def reiniciar(self):
        self.timer.stop()
        self.en_marcha = False
        self.inicializada = False
        self.reiniciar_al_iniciar = False
        self.n = 0
        self.U = []
        self.R = []
        self.boton_simulacion.label.set_text("Iniciar simulacion")
        self.estado.set_text("Parametros listos. Pulsa Iniciar simulacion.")
        self.limpiar_graficas()
        self.fig.canvas.draw_idle()

    def limpiar_graficas(self):
        for axis in (self.ax_omega, self.ax_error, self.ax_fase, self.ax_actitud):
            axis.clear()
        self.ax_omega.set(xlabel="Tiempo", ylabel="Velocidad angular")
        self.ax_error.set(xlabel="Tiempo", ylabel="Error relativo")
        self.ax_fase.set(
            xlabel=r"$\omega_1$",
            ylabel=r"$\omega_2$",
            zlabel=r"$\omega_3$",
        )
        self.ax_actitud.set(xlabel="X", ylabel="Y", zlabel="Z")

    def preparar_simulacion(self):
        self.limpiar_graficas()
        self.I1, self.I2, self.I3 = [
            self.sliders[f"I{n}"].val for n in range(1, 4)
        ]
        self.W0 = np.array(
            [self.sliders[f"w{n}(0)"].val for n in range(1, 4)]
        )
        self.tf = self.sliders["tf"].val
        self.N = int(np.ceil(self.tf / self.sliders["dt"].val))
        self.t = np.linspace(0, self.tf, self.N + 1)
        self.At = self.t[1] - self.t[0]
        self.F = lambda W, t: ecuaciones_rotacion(W, self.I1, self.I2, self.I3)
        self.J = lambda W, t: jacobiano(W, self.I1, self.I2, self.I3)
        self.mallas = [[self.W0.copy()] for _ in range(3)]
        self.U = self.mallas[0]
        self.R = [np.eye(3)]
        self.n = 0

        energia, momento = historial_invariantes(
            self.W0.reshape(1, 3), self.I1, self.I2, self.I3
        )
        self.energia0 = energia[0]
        self.momento0 = momento[0]
        self.dibujar_superficies()

        self.lineas_omega = []
        for k in range(3):
            (linea,) = self.ax_omega.plot([0], [self.W0[k]], label=fr"$\omega_{k + 1}$")
            self.lineas_omega.append(linea)
        self.ax_omega.set(
            xlim=(0, self.tf),
            xlabel="Tiempo",
            ylabel="Velocidad angular",
            title=f"Esquema {self.esquema}",
        )
        self.ax_omega.grid()
        self.ax_omega.legend()

        self.linea_energia, = self.ax_error.semilogy([0], [1e-16], label="Energia")
        self.linea_momento, = self.ax_error.semilogy(
            [0], [1e-16], label=r"$|L|^2$"
        )
        self.ax_error.set(
            xlim=(0, self.tf),
            xlabel="Tiempo",
            ylabel="Error relativo",
            title="Invariantes",
        )
        self.ax_error.grid()
        self.ax_error.legend()

        (self.linea_fase,) = self.ax_fase.plot(
            [self.W0[0]], [self.W0[1]], [self.W0[2]], label="Trayectoria"
        )
        self.ax_fase.legend()

        colores = ("r", "g", "b")
        nombres = ("eje x", "eje y", "eje z")
        self.lineas_actitud = []
        for color, nombre in zip(colores, nombres):
            (linea,) = self.ax_actitud.plot(
                [0, self.R[0][0, len(self.lineas_actitud)]],
                [0, self.R[0][1, len(self.lineas_actitud)]],
                [0, self.R[0][2, len(self.lineas_actitud)]],
                color=color,
                label=nombre,
            )
            self.lineas_actitud.append(linea)
        self.ax_actitud.set(
            xlim=(-1, 1),
            ylim=(-1, 1),
            zlim=(-1, 1),
            xlabel="X",
            ylabel="Y",
            zlabel="Z",
            title="Actitud del satelite",
        )
        self.ax_actitud.legend()

        _, _, eigenvalues = estabilidad_ejes(self.I1, self.I2, self.I3)
        stable, amplification = stability_report(
            self.esquema, eigenvalues["I2"], self.At
        )
        self.estado.set_text(
            f"t = 0 / {self.tf:.2f}; estabilidad lineal del eje intermedio: "
            f"{'estable' if stable else 'inestable'} (|R|max={amplification:.3f})"
        )
        self.inicializada = True
        self.fig.canvas.draw_idle()

    def dibujar_superficies(self):
        self.ax_fase.clear()
        theta, phi = np.meshgrid(
            np.linspace(0, np.pi, 24), np.linspace(0, 2 * np.pi, 36)
        )
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        self.ax_fase.plot_surface(
            x * np.sqrt(2 * self.energia0 / self.I1),
            y * np.sqrt(2 * self.energia0 / self.I2),
            z * np.sqrt(2 * self.energia0 / self.I3),
            color="tab:blue",
            alpha=0.12,
        )
        self.ax_fase.plot_surface(
            x * np.sqrt(self.momento0) / self.I1,
            y * np.sqrt(self.momento0) / self.I2,
            z * np.sqrt(self.momento0) / self.I3,
            color="tab:green",
            alpha=0.12,
        )
        curva = separatriz(
            self.I1, self.I2, self.I3, self.energia0
        )
        self.ax_fase.plot(
            curva[:, 0], curva[:, 1], curva[:, 2], "r--", label="Separatriz"
        )
        self.ax_fase.set(
            xlabel=r"$\omega_1$",
            ylabel=r"$\omega_2$",
            zlabel=r"$\omega_3$",
            title="Espacio de fases",
        )

    def iniciar_pausar(self, _=None):
        if self.en_marcha:
            self.timer.stop()
            self.en_marcha = False
            self.reiniciar_al_iniciar = True
            self.boton_simulacion.label.set_text("Iniciar simulacion")
            self.estado.set_text(
                f"Simulacion pausada en t = {self.t[self.n]:.3f}. "
                "Al iniciar de nuevo, comenzara desde el principio."
            )
        else:
            if (
                not self.inicializada
                or self.n == self.N
                or self.reiniciar_al_iniciar
            ):
                self.preparar_simulacion()
                self.reiniciar_al_iniciar = False
            self.en_marcha = True
            self.boton_simulacion.label.set_text("Pausar simulacion")
            self.timer.start()
        self.fig.canvas.draw_idle()

    def avanzar_paso(self):
        actual = self.U[-1].copy()
        for factor, malla in zip((1, 2, 4), self.mallas):
            dt = self.At / factor
            for subpaso in range(factor):
                indice = len(malla) - 1
                tiempo = self.t[self.n] + subpaso * dt
                if self.esquema == "LF" and indice == 0:
                    siguiente = RK4(malla[-1], dt, tiempo, self.F)
                else:
                    anterior = malla[-2] if self.esquema == "LF" else None
                    siguiente = paso_Cauchy(
                        self.esquema,
                        malla[-1],
                        anterior,
                        dt,
                        tiempo,
                        self.F,
                        self.J,
                    )
                malla.append(siguiente)
        nuevo = self.U[-1]

        rotation = paso_actitud(self.R[-1], actual, nuevo, self.At)
        self.R.append(rotation)
        self.n += 1

        estados = np.asarray(self.U)
        tiempos = self.t[: self.n + 1]
        energia, momento = historial_invariantes(
            estados, self.I1, self.I2, self.I3
        )
        error_energia = (
            abs(energia / self.energia0 - 1)
            if self.energia0 != 0
            else abs(energia - self.energia0)
        )
        error_momento = (
            abs(momento / self.momento0 - 1)
            if self.momento0 != 0
            else abs(momento - self.momento0)
        )
        for k, linea in enumerate(self.lineas_omega):
            linea.set_data(tiempos, estados[:, k])
        self.linea_energia.set_data(
            tiempos, np.maximum(error_energia, 1e-16)
        )
        self.linea_momento.set_data(
            tiempos, np.maximum(error_momento, 1e-16)
        )
        self.linea_fase.set_data(estados[:, 0], estados[:, 1])
        self.linea_fase.set_3d_properties(estados[:, 2])
        for k, linea in enumerate(self.lineas_actitud):
            eje = rotation[:, k]
            linea.set_data([0, eje[0]], [0, eje[1]])
            linea.set_3d_properties([0, eje[2]])

        self.estado.set_text(
            f"t = {tiempos[-1]:.3f} / {self.tf:.3f}; "
            f"error energia = {error_energia[-1]:.2e}; "
            f"error |L|^2 = {error_momento[-1]:.2e}"
        )
        self.fig.canvas.draw_idle()

        if self.n == self.N:
            self.timer.stop()
            self.en_marcha = False
            self.boton_simulacion.label.set_text("Reiniciar simulacion")
            gruesa, fina, mas_fina = [np.asarray(malla) for malla in self.mallas]
            diferencia1 = np.linalg.norm(gruesa - fina[::2])
            diferencia2 = np.linalg.norm(fina[::2] - mas_fina[::4])
            if diferencia2 == 0:
                self.estado.set_text(
                    f"Simulacion completada en t = {tiempos[-1]:.3f}; "
                    "solucion estacionaria, orden Richardson no definido."
                )
            else:
                orden = np.log(diferencia1 / diferencia2) / np.log(2)
                error = (gruesa - fina[::2]) / (2**orden - 1)
                self.estado.set_text(
                    f"Simulacion completada en t = {tiempos[-1]:.3f}; "
                    f"orden Richardson = {orden:.2f}; "
                    f"error estimado final = {np.linalg.norm(error[-1]):.2e}"
                )
        return True


def main():
    RotationDashboard()
    plt.show()


if __name__ == "__main__":
    main()
