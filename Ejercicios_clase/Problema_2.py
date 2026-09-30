import matplotlib.pyplot as plt
from numpy import array, zeros
# Comentaddo con Copilot
# Modelo físico: oscilador armónico simple.
# La variable de estado es U = [x, v], donde:
#   x = posición
#   v = velocidad
# Las ecuaciones son:
#   x' = v
#   v' = -x
# Esto equivale a x'' + x = 0, es decir, un movimiento armónico.


def F(U):
    # Derivada del estado U = [x, v]
    return array([U[1], -U[0]])

# Número total de pasos de tiempo
N = 1000000
# Tamaño del paso temporal
At = 0.1
# Número de componentes del vector de estado
Nv = 2

# Matriz que guarda la evolución del sistema:
# U[n,0] = x_n, U[n,1] = v_n
U = zeros((N + 1, Nv))

# Condición inicial: x(0)=1, v(0)=0
U[0, :] = array([1, 0])

# Integración numérica con el método de Runge-Kutta de cuarto orden (RK4).
# Este método aproxima mejor la solución que el método de Euler.
# En cada paso calculamos cuatro pendientes:
#   k1 = f(U_n)
#   k2 = f(U_n + At*k1/2)
#   k3 = f(U_n + At*k2/2)
#   k4 = f(U_n + At*k3)
# Luego se hace la combinación ponderada:
#   U_{n+1} = U_n + At*(k1 + 2*k2 + 2*k3 + k4)/6
for n in range(0, N):
    k1 = F(U[n, :])
    k2 = F(U[n, :] + At * k1 / 2)
    k3 = F(U[n, :] + At * k2 / 2)
    k4 = F(U[n, :] + At * k3)
    U[n + 1, :] = U[n, :] + At * (k1 + 2 * k2 + 2 * k3 + k4) / 6

# Graficamos la trayectoria en el plano (x, v).
# Para un oscilador armónico, esta gráfica da una órbita cerrada, normalmente circular.
plt.plot(U[:, 0], U[:, 1])
plt.axis('equal')
plt.xlabel('Posición x')
plt.ylabel('Velocidad v')
plt.title('Órbita del oscilador armónico (RK4)')
plt.show()
