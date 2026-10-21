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
import Cauchy_Problem
import Graphic

N_t = 100 #Numero de pasos de tiempo para la integración
t_i = 0
t_f = 10
U0 = array([1, 0])
#ordenes = {"EE": 1, "EI": 1, "CN": 2, "RK4": 4}
esquemas = ["EE", "EI", "CN", "RK4"]
soluciones = {}
errores = {}
for esquema in esquemas:
#for esquema, orden in ordenes.items():
    coarse, error, orden = Cauchy_Problem.sol_error_cauchy(esquema, U0, N_t, t_i, t_f, F)
    print(f"Esquema: {esquema}, Orden estimado: {orden:.2f}")
    soluciones[esquema] = coarse
    errores[esquema] = error

tiempos = linspace(t_i, t_f, N_t + 1)

Graphic.grafico_sol(soluciones)
Graphic.grafico_error(tiempos, errores)
plt.show()
