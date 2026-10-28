import numpy as np
import matplotlib.pyplot as plt

# Ecuación: y' = y, y(0) = 1
f = lambda t, y: y
y_exact = lambda t: np.exp(t)

# Punto inicial
tn = 0.0
yn = 1.0
h = 0.6

# Euler explícito
tn1 = tn + h
yn1 = yn + h * f(tn, yn)

# Solución exacta
y_real = y_exact(tn1)

# Curva exacta
t = np.linspace(0, 1.0, 300)
y = y_exact(t)

# Recta tangente (Euler)
tt = np.linspace(tn, tn1, 100)
ytan = yn + f(tn, yn) * (tt - tn)

plt.figure(figsize=(8,5))

# Solución exacta
plt.plot(t, y, 'b', lw=2, label='Solución exacta')

# Tangente de Euler
plt.plot(tt, ytan, 'r--', lw=2, label='Euler explícito')

# Puntos
plt.plot(tn, yn, 'ko', ms=8)
plt.plot(tn1, yn1, 'ro', ms=8, label=r'$y_{n+1}$ numérico')
plt.plot(tn1, y_real, 'bo', ms=8, label=r'$y(t_{n+1})$ exacto')

# Flecha pendiente
plt.annotate('Pendiente $f(t_n,y_n)$',
             xy=(0.25, 1.25),
             xytext=(0.05, 1.6),
             arrowprops=dict(arrowstyle='->'))

# Segmento de error
plt.plot([tn1, tn1], [yn1, y_real], 'k:', lw=1.5)
plt.text(tn1 + 0.02, (yn1+y_real)/2, 'error local')

plt.xlabel('t')
plt.xticks([tn, tn1], ['N', 'N+1'])
plt.ylabel('y')
plt.title('Método de Euler explícito')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.show()