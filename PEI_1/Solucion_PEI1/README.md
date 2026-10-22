# Solucion PEI 1

Solucion directa del problema de rotacion libre del solido rigido.

- `space_physics.py`: ecuaciones de Euler, Jacobiano, invariantes, estabilidad
  de los ejes, separatriz y orientacion del satelite.
- `numerical_engine.py`: EE, EI, Crank-Nicolson, RK4, Leapfrog y Richardson.
- `gui_reactive.py`: GUI con deslizadores, seleccion de esquema y boton para
  iniciar y pausar la simulacion paso a paso. Al pausar, las graficas se
  mantienen visibles; al volver a iniciar, se limpian y el calculo comienza
  desde el estado inicial.
- `cuerpos.py`: catalogo de cuerpos (raqueta, mango en T, satelite, libro,
  asteroide, cohete). Cada uno se construye con piezas homogeneas y de ahi se
  calcula su tensor de inercia; x, y, z son los ejes de I1 < I2 < I3.
- `gui_actitud.py`: cuadro de mando con el cuerpo 3D girando en el espacio
  inercial (L vertical), sus ejes principales, L y omega; selector de cuerpo,
  de eje de giro inicial y de integrador; espacio de fases con las superficies
  de Poinsot y la separatriz; omega(t); deriva de 2T y L^2; analisis espectral
  de los tres ejes, |R(z)| del esquema y error de Richardson.

## Ejecutar

Desde la carpeta principal del proyecto:

```powershell
.\.venv\Scripts\python.exe .\PEI_1\Solucion_PEI1\gui_actitud.py
.\.venv\Scripts\python.exe .\PEI_1\Solucion_PEI1\gui_reactive.py
```

En `gui_actitud.py`: barra espaciadora para iniciar/pausar y arrastrar el
raton sobre una vista 3D para girar la camara. Al elegir un cuerpo se cargan
sus momentos de inercia; si se mueven los deslizadores I1, I2, I3 el cuerpo
se deforma para tener exactamente esos momentos.

## Pruebas

```powershell
.\.venv\Scripts\python.exe -m unittest PEI_1.Solucion_PEI1.test_solution -v
```
