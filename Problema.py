import matplotlib.pyplot as plt
from numpy import array, zeros
from numpy.linalg import norm

# Modelo físico: oscilador armónico simple.
# Se representa el estado del sistema con U = [x, v], donde:
#   x = posición
#   v = velocidad
# La dinámica viene dada por:
#   x' = v
#   v' = -x
# equivalente a x'' + x = 0.


def F(U):
    # Derivada del vector de estado U = [x, v]
    return array([U[1], -U[0]])

# Número de pasos temporales
N = 100
# Tamaño de paso de integración
At = 0.1
# Número de variables del sistema
Nv = 2

# Matriz para guardar la evolución temporal del sistema:
# U[n,0] = x_n, U[n,1] = v_n
U = zeros((N + 1, Nv))

# Condición inicial: x(0) = 1, v(0) = 0
U[0, :] = array([1, 0])

# Integración con un método implícito de punto medio.
# Se desea encontrar Y tal que:
#   Y = U[n] + At/2 * (F(U[n]) + F(Y))
# Es decir, la pendiente se toma como promedio entre el inicio y el fin del paso.
# Se resuelve iterativamente hasta que la diferencia residual es muy pequeña.
for n in range(0, N):
    Y = U[n, :]
    while norm(Y - U[n, :] - At / 2 * (F(U[n, :]) + F(Y))) > 1e-6:
        R = Y - U[n, :] - At / 2 * (F(U[n, :]) + F(Y))
        Y = Y + R
        print(norm(R))
        input("Press Enter to continue...")
    U[n + 1, :] = Y

# Se grafican las trayectorias en el plano (x, v), que para un oscilador armónico
# describen una órbita cerrada alrededor del origen.
plt.plot(U[:, 0], U[:, 1])
plt.axis('equal')
plt.xlabel('Posición x')
plt.ylabel('Velocidad v')
plt.title('Órbita del oscilador armónico (método implícito)')
plt.show()

