"""
Ejemplo de esquemas de integración temporal para el oscilador armónico simple.
El error temporal de cada método se estima con Richardson, comparando los
resultados con pasos h y h/2.
"""
import matplotlib.pyplot as plt
from numpy import abs, array, linspace

from Integradores import integrar_Cauchy
from Richardson import error_richardson
from Esquema_fisico import oscilador as F

N_t = 100 #Numero de pasos de tiempo para la integración
t_i = 0
t_f = 10
U0 = array([1, 0])
ordenes = {"EE": 1, "EI": 1, "CN": 2, "RK4": 4}
soluciones = {}
errores = {}

for esquema, orden in ordenes.items():
    coarse = integrar_Cauchy(esquema, U0, N_t, t_i, t_f, F)
    fine = integrar_Cauchy(esquema, U0, 2 * N_t, t_i, t_f, F)
    soluciones[esquema] = coarse
    errores[esquema] = abs(error_richardson(coarse, fine[::2], orden))

tiempos = linspace(t_i, t_f, N_t + 1)

for esquema, solucion in soluciones.items():
    plt.plot(solucion[:, 0], solucion[:, 1], label=esquema)
plt.axhline(0, color="black", linewidth=0.8)
plt.axvline(0, color="black", linewidth=0.8)
plt.scatter([1, -1], [0, 0], color="black", zorder=5)
plt.annotate("(1, 0)", (1, 0), xytext=(6, 6), textcoords="offset points")
plt.annotate("(-1, 0)", (-1, 0), xytext=(6, 6), textcoords="offset points")
plt.axis("equal")
plt.xlabel("Posición x")
plt.ylabel("Velocidad v")
plt.title("Órbita del oscilador armónico simple")
plt.legend()

plt.figure()
for esquema, error in errores.items():
    plt.plot(tiempos, error[:, 0], label=f"Error en x ({esquema})")
    plt.plot(tiempos, error[:, 1], label=f"Error en v ({esquema})")
plt.xlabel("Tiempo")
plt.ylabel("Error")
plt.title("Error temporal estimado mediante Richardson")
plt.legend()
plt.show()
