from numpy import array

def oscilador(U:array, t:float):
    """Ecuación del oscilador armónico simple.
    U = [x, v], entonces el sistema se define como:
        dx/dt = v
        dv/dt = -x
    Esto corresponde a la ecuación diferencial x'' + x = 0.
    """
    # Estado del sistema: U[0] = posición, U[1] = velocidad.    
    return array([U[1], -U[0]])