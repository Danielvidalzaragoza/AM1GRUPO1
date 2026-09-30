import matplotlib.pyplot as plt
from numpy import array, zeros
from numpy.linalg import norm

# Comentado con Copilot
# Modelo físico: oscilador armónico simple.
# La variable de estado es U = [x, v], donde:
#   x = posición
#   v = velocidad
# La ecuación es:
#   x' = v
#   v' = -x
# o equivalentemente x'' + x = 0.


def F(U):
    # Derivada del estado U = [x, v]
    return array([U[1], -U[0]])

# Número de pasos temporales
N = 100
# Tamaño del paso temporal
At = 0.1
# Número de componentes del vector de estado
Nv = 2

# Matriz que guarda la evolución del sistema en el tiempo:
# U[n,0] = x_n, U[n,1] = v_n
U = zeros((N + 1, Nv))

# Condición inicial: x(0)=1, v(0)=0
U[0, :] = array([1, 0])

# Integración con el método implícito del punto medio.
# En cada paso se busca Y tal que:
#   Y = U[n] + At/2 * (F(U[n]) + F(Y))
# Es decir, se usa la media de la derivada en el inicio y en el final del intervalo.
# Esto produce un método más estable que Euler para ciertos sistemas.
for n in range(0, N):
    # Y es el valor aproximado del estado en el siguiente paso temporal.
    # Se inicializa con el valor actual para luego corregirlo iterativamente.
    Y = U[n, :]

    # Mientras no se cumpla la ecuación implícita con la precisión pedida,
    # seguimos ajustando Y.
    # La expresión dentro de norm(...) es el residuo de la ecuación:
    #   R = Y - U[n] - (At/2) * (F(U[n]) + F(Y))
    # Cuando R ≈ 0, entonces Y satisface el método del punto medio implícito.
    while norm(Y - U[n, :] - At / 2 * (F(U[n, :]) + F(Y))) > 1e-10:
        # Calculamos el residuo residual para saber cuánto falta para cumplir la ecuación.
        R = Y - U[n, :] - At / 2 * (F(U[n, :]) + F(Y))

        # Corregimos Y restando el residuo. Esto mueve Y hacia la solución de la ecuación implícita.
        # Si R es positivo, disminuimos Y; si es negativo, aumentamos Y.
        Y = Y - R

        # Muestra la magnitud del error residual en cada iteración.
        # Sirve para ver cómo converge la solución.
        print(norm(R))
        # input("Press Enter to continue...")

    # Cuando el residuo es suficientemente pequeño, aceptamos Y como el siguiente estado.
    U[n + 1, :] = Y

# Graficamos la trayectoria en el plano (x, v), que para este sistema es una órbita cerrada.
plt.plot(U[:, 0], U[:, 1])
plt.axis('equal')
plt.xlabel('Posición x')
plt.ylabel('Velocidad v')
plt.title('Órbita del oscilador armónico (punto medio implícito)')
plt.show()

