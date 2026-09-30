import matplotlib.pyplot as plt
from numpy import array, zeros

# Modelo físico: oscilador armónico simple sin amortiguamiento.
# La variable U = [x, v] representa posición x y velocidad v.
# La ecuación del sistema es:
#   x' = v
#   v' = -x
# que corresponde a x'' + x = 0.
# Esto describe un movimiento circular/armónico en el plano (x, v).

def F(U):
    # Derivada del estado U = [x, v]
    return array([U[1], -U[0]])

# Número de pasos de tiempo
N = 10000
# Paso de integración (tiempo discreto)
At = 0.001
# Número de variables del estado
Nv = 2

# Matriz que guarda el estado en cada instante de tiempo:
# U[n,0] = x_n, U[n,1] = v_n
U = zeros((N + 1, Nv))

# Condición inicial: posición inicial x = 1, velocidad inicial v = 0
U[0, :] = array([1, 0])

# Integración explícita de Euler:
# U[n+1] = U[n] + Δt * F(U[n])
# Esta es la versión discreta de la ecuación de evolución del sistema.
for n in range(0, N):
    U[n + 1, :] = U[n, :] + At * F(U[n, :])

# Graficamos la trayectoria en el plano (x, v), que para este sistema
# es una órbita circular alrededor del origen.
plt.plot(U[:, 0], U[:, 1])
plt.axis('equal')
plt.xlabel('Posición x')
plt.ylabel('Velocidad v')
plt.title('Órbita del oscilador armónico simple')
plt.show()
