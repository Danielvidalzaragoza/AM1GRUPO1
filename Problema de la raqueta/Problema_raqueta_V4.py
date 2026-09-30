from numpy import array, zeros, sqrt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ========================================
# 1. MOMENTOS PRINCIPALES DE INERCIA
# ========================================

I1 = 1.0
I2 = 2.0
I3 = 3.0


# ========================================
# 2. TIEMPO DE SIMULACIÓN
# ========================================

Dt = 0.001
N = 100000


# ========================================
# 3. ECUACIONES DE EULER DEL CUERPO RÍGIDO
# ========================================

def F(U):

    w1 = U[0]
    w2 = U[1]
    w3 = U[2]

    dw1 = ((I2 - I3) / I1) * w2 * w3
    dw2 = ((I3 - I1) / I2) * w3 * w1
    dw3 = ((I1 - I2) / I3) * w1 * w2

    return array([dw1, dw2, dw3])


# ========================================
# 4. OPERACIONES CON CUATERNIONES
# ========================================

def multiplicar_cuaterniones(q, p):

    q0, q1, q2, q3 = q
    p0, p1, p2, p3 = p

    return array([
        q0*p0 - q1*p1 - q2*p2 - q3*p3,
        q0*p1 + q1*p0 + q2*p3 - q3*p2,
        q0*p2 - q1*p3 + q2*p0 + q3*p1,
        q0*p3 + q1*p2 - q2*p1 + q3*p0
    ])


def matriz_rotacion(q):

    q0, q1, q2, q3 = q

    return array([
        [
            1 - 2*(q2*q2 + q3*q3),
            2*(q1*q2 - q0*q3),
            2*(q1*q3 + q0*q2)
        ],
        [
            2*(q1*q2 + q0*q3),
            1 - 2*(q1*q1 + q3*q3),
            2*(q2*q3 - q0*q1)
        ],
        [
            2*(q1*q3 - q0*q2),
            2*(q2*q3 + q0*q1),
            1 - 2*(q1*q1 + q2*q2)
        ]
    ])


# ========================================
# 5. GUARDAMOS LA SIMULACIÓN
# ========================================

U = zeros((N + 1, 3))

# Q guarda la orientación del cuerpo
Q = zeros((N + 1, 4))


# Giro casi alrededor del eje intermedio
U[0, :] = array([0.01, 1.0, 0.01])

# Orientación inicial
Q[0, :] = array([1.0, 0.0, 0.0, 0.0])


# ========================================
# 6. INTEGRACIÓN TEMPORAL
# ========================================

for n in range(0, N):

    # Velocidad angular: Euler explícito
    U[n + 1, :] = U[n, :] + Dt * F(U[n, :])

    # Velocidad angular en forma de cuaternión
    omega = array([
        0.0,
        U[n, 0],
        U[n, 1],
        U[n, 2]
    ])

    # Cómo cambia la orientación
    dQ = 0.5 * multiplicar_cuaterniones(Q[n, :], omega)

    # Euler para la orientación
    Q[n + 1, :] = Q[n, :] + Dt * dQ

    # Normalizamos el cuaternión
    norma = sqrt((Q[n + 1, :]**2).sum())

    Q[n + 1, :] = Q[n + 1, :] / norma


# ========================================
# 7. CUERPO QUE VAMOS A DIBUJAR
# ========================================

# Dimensiones del cuerpo
a = 3.0
b = 1.2
c = 0.25

vertices = array([
    [-a/2, -b/2, -c/2],
    [ a/2, -b/2, -c/2],
    [ a/2,  b/2, -c/2],
    [-a/2,  b/2, -c/2],

    [-a/2, -b/2,  c/2],
    [ a/2, -b/2,  c/2],
    [ a/2,  b/2,  c/2],
    [-a/2,  b/2,  c/2]
])


aristas = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]


# ========================================
# 8. PREPARAMOS LA GRÁFICA 3D
# ========================================

fig = plt.figure()

ax = fig.add_subplot(111, projection="3d")

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_box_aspect((1, 1, 1))


lineas = []

for arista in aristas:
    linea, = ax.plot([], [], [])
    lineas.append(linea)


# Dibujamos también los tres ejes del cuerpo

ejes = []

for i in range(3):
    linea, = ax.plot([], [], [], linewidth=3)
    ejes.append(linea)


# ========================================
# 9. ANIMACIÓN
# ========================================

salto = 200


def actualizar(frame):

    n = frame * salto

    R = matriz_rotacion(Q[n, :])

    # Giramos todos los vértices
    vertices_rotados = vertices @ R.T

    # Actualizamos el cuerpo
    for linea, arista in zip(lineas, aristas):

        i, j = arista

        x = [
            vertices_rotados[i, 0],
            vertices_rotados[j, 0]
        ]

        y = [
            vertices_rotados[i, 1],
            vertices_rotados[j, 1]
        ]

        z = [
            vertices_rotados[i, 2],
            vertices_rotados[j, 2]
        ]

        linea.set_data_3d(x, y, z)


    # Ejes propios del cuerpo
    longitud = 1.5

    for i in range(3):

        eje = R[:, i] * longitud

        ejes[i].set_data_3d(
            [0, eje[0]],
            [0, eje[1]],
            [0, eje[2]]
        )


    tiempo = n * Dt

    ax.set_title(f"t = {tiempo:.2f}")

    return lineas + ejes


numero_frames = N // salto


animacion = FuncAnimation(
    fig,
    actualizar,
    frames=numero_frames,
    interval=30,
    blit=False,
    repeat=True
)


plt.show()