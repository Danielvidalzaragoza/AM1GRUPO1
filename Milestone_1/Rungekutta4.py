from numpy import array, zeros
import matplotlib.pyplot as plt

N = 1000
Dt = 0.01

Nv = 2

U = zeros((N+1, Nv))
U[0, :] = array([1, 0])


def F(U: array) -> array:
    return array([U[1], -U[0]])


for n in range(0, N):
    k_1 = F(U[n, :])
    k_2 = F(U[n, :] + Dt * k_1 / 2)
    k_3 = F(U[n, :] + Dt * k_2 / 2)
    k_4 = F(U[n, :] + Dt * k_3)
    U[n + 1, :] = U[n, :] + Dt * (k_1 + 2 * k_2 + 2 * k_3 + k_4) / 6
plt.plot(U[:, 0], U[:, 1])
plt.axis('equal')
plt.show()