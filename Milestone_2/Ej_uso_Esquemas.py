#ejemplo
import matplotlib.pyplot as plt
from numpy import array, zeros
from Esquemas_temporales import EE

def F(U:array, t:float):
    # Derivada del estado U = [x, v]
    return array([U[1], -U[0]])
N = 100
At = 0.1
Nv = 2
U = zeros((N + 1, Nv))
U[0, :] = array([1, 0])
for n in range(0, N):
    U[n + 1, :] =EE(U[n, :], At, n*At, F)

plt.plot(U[:, 0], U[:, 1])
plt.axis('equal')
plt.xlabel('Posición x')
plt.ylabel('Velocidad v')
plt.title('Órbita del oscilador armónico simple')
plt.show()