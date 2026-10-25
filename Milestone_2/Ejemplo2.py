''' 
ejemplo del uso de los esquemas de integración temporal y funcion de ingegrador de Cauchy
 para el oscilador armónico simple
se calculan los errores del EI, CN y RK4 con respecto al EE, 
que es el esquema más simple y rápido, pero menos preciso.
'''

import matplotlib.pyplot as plt
from numpy import array, zeros, abs
from Esquemas_temporales import EE,RK4,EI,CN
from Integradores import integrar_Cauchy
from Esquema_fisico import oscilador as F

N_t = 100 #Numero de pasos de tiempo
Nv = 2 #dimesiones del problema (x,v)
t_i=0 #tiempo inicial
t_f=10 #tiempo final
U_1 = zeros((N_t + 1, Nv))
U_1[0, :] = array([1, 0])
U_2 = zeros((N_t + 1, Nv))
U_2[0, :] = array([1, 0])
U_3 = zeros((N_t + 1, Nv))
U_3[0, :] = array([1, 0])
U_4 = zeros((N_t + 1, Nv))
U_4[0, :] = array([1, 0])

U_1 = integrar_Cauchy("EE", U_1[0, :] , N_t, t_i, t_f, F)
U_2 = integrar_Cauchy("RK4", U_2[0, :] , N_t, t_i, t_f, F)
U_3 = integrar_Cauchy("EI", U_3[0, :] , N_t, t_i, t_f, F)
U_4 = integrar_Cauchy("CN", U_4[0, :] , N_t, t_i, t_f, F)

ERR_RK4 = abs(U_1 - U_2)
ERR_EI = abs(U_1 - U_3)
ERR_CN = abs(U_1 - U_4)
plt.plot(U_1[:, 0], U_1[:, 1], label='EE')
plt.plot(U_2[:, 0], U_2[:, 1], label='RK4')
plt.plot(U_3[:, 0], U_3[:, 1], label='EI')
plt.plot(U_4[:, 0], U_4[:, 1], label='CN')
plt.axis('equal')
plt.xlabel('Posición x')
plt.ylabel('Velocidad v')
plt.title('Órbita del oscilador armónico simple')
plt.legend()
plt.figure()
plt.plot(ERR_RK4[:, 0], label='Error en x (RK4)')
plt.plot(ERR_RK4[:, 1], label='Error en v (RK4)')
plt.plot(ERR_EI[:, 0], label='Error en x (EI)')
plt.plot(ERR_EI[:, 1], label='Error en v (EI)')
plt.plot(ERR_CN[:, 0], label='Error en x (CN)')
plt.plot(ERR_CN[:, 1], label='Error en v (CN)')
plt.xlabel('Paso de tiempo')
plt.ylabel('Error')
plt.title('Comparación de esquemas de integración temporal con Euler Explícito')
plt.legend()
plt.show()
