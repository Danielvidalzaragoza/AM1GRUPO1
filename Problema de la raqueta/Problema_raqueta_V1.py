from numpy import array, zeros
import matplotlib.pyplot as plt


# Momentos principales de inercia
I1 = 1.0
I2 = 2.0
I3 = 3.0


# Paso temporal
Dt = 0.001

# Número de pasos
N = 20000

# Número de variables del estado
Nv = 3


def F(U):
    w1 = U[0]
    w2 = U[1]
    w3 = U[2]

    dw1 = ((I2 - I3) / I1) * w2 * w3
    dw2 = ((I3 - I1) / I2) * w3 * w1
    dw3 = ((I1 - I2) / I3) * w1 * w2

    return array([dw1, dw2, dw3])


# Tabla donde guardamos toda la simulación
U = zeros((N + 1, Nv))

# Condición inicial:
# gira principalmente alrededor del eje intermedio
U[0, :] = array([0.01, 1.0, 0.01])


# Método de Euler
for n in range(0, N):
    U[n + 1, :] = U[n, :] + Dt * F(U[n, :])


# Vector de tiempos
t = zeros(N + 1)

for n in range(0, N):
    t[n + 1] = t[n] + Dt


# Dibujamos las tres velocidades angulares
plt.plot(t, U[:, 0], label="omega 1")
plt.plot(t, U[:, 1], label="omega 2")
plt.plot(t, U[:, 2], label="omega 3")

plt.xlabel("tiempo")
plt.ylabel("velocidad angular")
plt.legend()
plt.show()