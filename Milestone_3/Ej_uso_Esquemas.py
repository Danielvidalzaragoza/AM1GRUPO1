"""Ejemplo de esquemas de integración temporal para el oscilador armónico simple.
"""
import matplotlib.pyplot as plt
from numpy import abs, array, linspace
from Integradores import integrar_Cauchy
from Richardson import error_richardson
from Esquema_fisico import oscilador as F
import Cauchy_Problem
import Graphic
import Convergencia

N_t = 5000 #Numero de pasos de tiempo para la integración
t_i = 0
t_f = 10
U0 = array([1, 0])
esquemas = ["EE", "EI", "CN", "RK4", "LF"]
soluciones = {}
errores = {}
error_final_por_esquema = {}
pasos = {}
for esquema in esquemas:
    coarse, error = Cauchy_Problem.sol_error_cauchy(esquema, U0, N_t, t_i, t_f, F)
    #print(f"Esquema: {esquema}, Orden estimado: {orden:.2f}")
    soluciones[esquema] = coarse
    errores[esquema] = error
    error_final_por_esquema[esquema],pasos[esquema] = Convergencia.converge_sys(esquema, U0, t_i, t_f, F)

tiempos = linspace(t_i, t_f, N_t + 1)

plt.close("all")
Graphic.grafico_sol(soluciones)
Graphic.grafico_error(tiempos, errores)
Graphic.grafico_convergencia(error_final_por_esquema, pasos)
plt.show()
