import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")

from .gui_reactive import RotationDashboard
from .numerical_engine import integrar_Cauchy, richardson_error, stability_report
from .space_physics import (
    actitud,
    ecuaciones_rotacion,
    estabilidad,
    estabilidad_ejes,
    historial_invariantes,
    invariantes,
    jacobiano,
    separatriz,
)


class PruebasPEI1(unittest.TestCase):
    def setUp(self):
        self.I1, self.I2, self.I3 = 1.0, 2.0, 3.0
        self.F = lambda W, t: ecuaciones_rotacion(
            W, self.I1, self.I2, self.I3
        )
        self.J = lambda W, t: jacobiano(W, self.I1, self.I2, self.I3)

    def test_ecuaciones_y_jacobiano(self):
        W = np.array([0.2, 0.8, -0.3])
        self.assertTrue(
            np.allclose(self.F(W, 0), [0.24, -0.06, -0.0533333333333])
        )
        h = 1e-6
        Jnum = np.column_stack(
            [
                (self.F(W + h * np.eye(3)[i], 0) - self.F(W - h * np.eye(3)[i], 0))
                / (2 * h)
                for i in range(3)
            ]
        )
        self.assertTrue(np.allclose(self.J(W, 0), Jnum))

    def test_estabilidad_de_los_ejes(self):
        _, _, eigenvalues = estabilidad_ejes(self.I1, self.I2, self.I3)
        self.assertTrue(np.any(np.abs(eigenvalues["I1"].imag) > 0))
        self.assertTrue(np.any(np.abs(eigenvalues["I2"].real) > 0))
        self.assertTrue(np.any(np.abs(eigenvalues["I3"].imag) > 0))

    def test_integradores_y_invariantes(self):
        W0 = np.array([0.1, 0.9, 0.2])
        for method in ("EE", "EI", "CN", "RK4", "LF"):
            with self.subTest(method=method):
                t, U = integrar_Cauchy(method, W0, 1000, 0, 2, self.F, self.J)
                self.assertEqual(U.shape, (len(t), 3))
                energia, momento = historial_invariantes(
                    U, self.I1, self.I2, self.I3
                )
                self.assertTrue(np.all(np.isfinite(energia)))
                self.assertTrue(np.all(np.isfinite(momento)))

    def test_separatriz_y_richardson(self):
        energia, _ = invariantes([0.1, 1.0, 0.1], self.I1, self.I2, self.I3)
        curva = separatriz(self.I1, self.I2, self.I3, energia)
        energia_curva, momento_curva = historial_invariantes(
            curva, self.I1, self.I2, self.I3
        )
        self.assertTrue(np.allclose(energia_curva, energia))
        self.assertTrue(np.allclose(momento_curva, 2 * self.I2 * energia))
        t, U, error, order = richardson_error(
            "RK4", [1.0], 200, 0, 1, lambda W, t: W, lambda W, t: np.eye(1)
        )
        self.assertEqual(U.shape, error.shape)
        self.assertGreater(order, 3.5)

    def test_funciones_de_estabilidad(self):
        self.assertTrue(np.allclose(estabilidad("RK4", [0]), [1]))
        stable, _ = stability_report("RK4", [1j], 0.1)
        self.assertTrue(stable)

    def test_actitud(self):
        t = np.linspace(0, 2, 101)
        U = np.tile([0.0, 0.0, 1.0], (len(t), 1))
        R = actitud(U, t)
        self.assertTrue(
            np.allclose(R @ np.transpose(R, (0, 2, 1)), np.eye(3))
        )
        self.assertAlmostEqual(np.linalg.det(R[-1]), 1.0)

    def test_pausar_conserva_graficas_y_reinicio_las_limpia(self):
        dashboard = RotationDashboard()
        dashboard.sliders["dt"].set_val(0.08)
        dashboard.sliders["tf"].set_val(1.0)
        dashboard.iniciar_pausar()
        dashboard.avanzar_paso()
        omega_before_pause = [
            line.get_xdata().copy() for line in dashboard.ax_omega.lines
        ]
        self.assertGreater(len(omega_before_pause[0]), 1)
        dashboard.iniciar_pausar()
        self.assertFalse(dashboard.en_marcha)
        self.assertTrue(dashboard.inicializada)
        self.assertEqual(dashboard.boton_simulacion.label.get_text(), "Iniciar simulacion")
        self.assertEqual(len(dashboard.ax_omega.lines), 3)
        for line, xdata in zip(dashboard.ax_omega.lines, omega_before_pause):
            self.assertTrue(np.array_equal(line.get_xdata(), xdata))

        dashboard.iniciar_pausar()
        self.assertEqual(dashboard.n, 0)
        self.assertEqual(len(dashboard.U), 1)
        self.assertTrue(np.array_equal(dashboard.U[0], dashboard.W0))
        self.assertEqual(dashboard.boton_simulacion.label.get_text(), "Pausar simulacion")
        self.assertTrue(
            all(len(line.get_xdata()) == 1 for line in dashboard.ax_omega.lines)
        )
        while dashboard.n < dashboard.N:
            dashboard.avanzar_paso()
        self.assertEqual(dashboard.n, dashboard.N)
        self.assertEqual(
            dashboard.boton_simulacion.label.get_text(), "Reiniciar simulacion"
        )


if __name__ == "__main__":
    unittest.main()
