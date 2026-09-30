from numpy import array, zeros
import matplotlib.pyplot as plt

I1 = 1.0
I2 = 2.0
I3 = 3.0

Dt = 0.001
N = 100000
Nv = 3

def F(U):
    w1 = U[0]
    w2 = U[1]
    w3 = U[2]

    dw1 = ((I2 - I3) / I1) * w2 * w3
    dw2 = ((I3 - I1) / I2) * w3 * w1
    dw3 = ((I1 - I2) / I3) * w1 * w2

    return array([dw1, dw2, dw3])

U = zeros((N + 1, Nv))

# Giro casi alrededor del eje 1
U[0, :] = array([1.0, 0.01, 0.01])

for n in range(0, N):
    U[n + 1, :] = U[n, :] + Dt * F(U[n, :])

t = zeros(N + 1)

for n in range(0, N):
    t[n + 1] = t[n] + Dt

plt.plot(t, U[:, 0], label="omega 1")
plt.plot(t, U[:, 1], label="omega 2")
plt.plot(t, U[:, 2], label="omega 3")

plt.xlabel("tiempo")
plt.ylabel("velocidad angular")
plt.legend()
plt.show()