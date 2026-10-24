import matplotlib.pyplot as plt
from numpy import array
import Convergencia
import Graphic
from Esquema_fisico import oscilador as F

t_i = 0
t_f = 10
U0 = array([1, 0])
error, pasos = Convergencia.converge_sys("RK4", U0, t_i, t_f, F)
Graphic.grafico_convergencia({"RK4": error}, {"RK4": pasos})
plt.show()