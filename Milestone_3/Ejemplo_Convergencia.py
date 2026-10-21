"""
Estudio de la tasa de convergencia de los esquemas temporales
en función del paso de tiempo (h).
"""
import numpy as np
import matplotlib.pyplot as plt
from Esquema_fisico import oscilador as F
from Integradores import integrar_Cauchy

# Parámetros del problema
t_i = 0.0
t_f = 10.0
U0 = np.array([1.0, 0.0])
esquemas = ["EE", "EI", "CN", "RK4"]

# Diferentes números de pasos (y por tanto, diferentes h)
N_values = [20, 40, 80, 160, 320, 640]
dt_values = [(t_f - t_i) / N for N in N_values]

# Solución de referencia muy precisa usando RK4 con un paso muy pequeño
N_ref = 2560
ref_solution = integrar_Cauchy("RK4", U0, N_ref, t_i, t_f, F)

errores_globales = {esquema: [] for esquema in esquemas}

for esquema in esquemas:
    for N in N_values:
        sol = integrar_Cauchy(esquema, U0, N, t_i, t_f, F)
        # Comparamos la solución con la de referencia interpolada (submuestreo)
        step_ratio = N_ref // N
        sol_ref_subsampled = ref_solution[::step_ratio]
        
        # Error norm 
        error = np.linalg.norm(sol[-1] - sol_ref_subsampled[-1])
        errores_globales[esquema].append(error)

# Graficar resultados en escala Log-Log
fig, ax = plt.subplots(figsize=(8, 6))

for esquema in esquemas:
    ax.loglog(dt_values, errores_globales[esquema], marker='o', label=f"Esquema {esquema}")

# Líneas de referencia para las pendientes teóricas (p = 1, 2, 4)
dt_ref = np.array(dt_values)
ax.loglog(dt_ref, dt_ref * 1e-1, 'k--', label="Orden 1 (O(dt))")
ax.loglog(dt_ref, (dt_ref**2) * 1e-1, 'k-.', label="Orden 2 (O(dt^2))")
ax.loglog(dt_ref, (dt_ref**4) * 1e-1, 'k:', label="Orden 4 (O(dt^4))")

ax.set_xlabel("Paso de tiempo (dt)")
ax.set_ylabel("Error global")
ax.set_title("Convergencia de los esquemas temporales con el paso de tiempo")
ax.grid(True, which="both", ls="--")
ax.legend()

plt.show()