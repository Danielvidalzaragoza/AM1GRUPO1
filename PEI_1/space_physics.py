import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from numpy import array 

def sort_I (I_1:float, I_2:float, I_3:float):
    I1=min(I_1, I_2, I_3)
    I3=max(I_1, I_2, I_3)
    I2=I_1 + I_2 + I_3 - I1 - I3
    return I1, I2, I3

def ecuaciones_rotacion(W:array, I1:float, I2:float, I3:float):
    W[3]=(I2-I3)/I1*W[1]*W[2]
    W[4]=(I3-I1)/I2*W[1]*W[3]
    W[5]=(I1-I2)/I3*W[2]*W[3]
    return W

def invariantes(W:array, I1:float, I2:float, I3:float):
    energia=(I1*W[1]**2 + I2*W[2]**2 + I3*W[3]**2)
    momento=(I1*W[1])**2 + (I2*W[2])**2 + (I3*W[3])**2
    if 2*I1*energia/2-np.sqrt(momento)>0 or 2*I3*energia/2-np.sqrt(momento)<0:
        print("Los invariantes no son consistentes, revise los valores de entrada.")
    return 