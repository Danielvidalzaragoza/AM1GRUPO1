import numpy as np
def N_exp(i:int):
    return 10**(i)
pasos_lin =np.array([i for i in range(0, 10)])
pasos= N_exp(pasos_lin)
print("pasos", pasos, pasos_lin)