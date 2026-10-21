# Solucion PEI 1

Solucion directa del problema de rotacion libre del solido rigido.

- `space_physics.py`: ecuaciones de Euler, Jacobiano, invariantes, estabilidad
  de los ejes, separatriz y orientacion del satelite.
- `numerical_engine.py`: EE, EI, Crank-Nicolson, RK4, Leapfrog y Richardson.
- `gui_reactive.py`: GUI con deslizadores, seleccion de esquema y boton para
  iniciar y pausar la simulacion paso a paso. Al pausar, las graficas se
  mantienen visibles; al volver a iniciar, se limpian y el calculo comienza
  desde el estado inicial.

## Ejecutar

Desde la carpeta principal del proyecto:

```powershell
.\.venv\Scripts\python.exe .\PEI_1\Solucion_PEI1\gui_reactive.py
```

## Pruebas

```powershell
.\.venv\Scripts\python.exe -m unittest PEI_1.Solucion_PEI1.test_solution -v
```
