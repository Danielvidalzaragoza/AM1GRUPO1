from numpy import array, zeros
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


# ==========================================================
# SIMULACIÓN
# ==========================================================

Dt = 0.002
Tfinal = 100.0

N = int(Tfinal / Dt)


def simular(I1, I2, I3, w10, w20, w30):

    # Tabla de estados:
    # columna 0 -> omega 1
    # columna 1 -> omega 2
    # columna 2 -> omega 3

    U = zeros((N + 1, 3))

    # Condiciones iniciales

    U[0, :] = array([w10, w20, w30])


    # Función F(U)

    def F(U):

        w1 = U[0]
        w2 = U[1]
        w3 = U[2]

        dw1 = ((I2 - I3) / I1) * w2 * w3
        dw2 = ((I3 - I1) / I2) * w3 * w1
        dw3 = ((I1 - I2) / I3) * w1 * w2

        return array([dw1, dw2, dw3])


    # Euler explícito

    for n in range(0, N):

        U[n + 1, :] = U[n, :] + Dt * F(U[n, :])


    # Vector de tiempos

    t = zeros(N + 1)

    for n in range(0, N):

        t[n + 1] = t[n] + Dt


    return t, U


# ==========================================================
# VALORES INICIALES
# ==========================================================

I1_inicial = 1.0
I2_inicial = 2.0
I3_inicial = 3.0

w1_inicial = 0.01
w2_inicial = 1.0
w3_inicial = 0.01


# Primera simulación

t, U = simular(
    I1_inicial,
    I2_inicial,
    I3_inicial,
    w1_inicial,
    w2_inicial,
    w3_inicial
)


# ==========================================================
# FIGURA
# ==========================================================

fig, ax = plt.subplots()

# Dejamos espacio abajo para los controles

plt.subplots_adjust(
    left=0.12,
    bottom=0.48
)


# Líneas

linea1, = ax.plot(
    t,
    U[:, 0],
    label="omega 1"
)

linea2, = ax.plot(
    t,
    U[:, 1],
    label="omega 2"
)

linea3, = ax.plot(
    t,
    U[:, 2],
    label="omega 3"
)


ax.set_xlabel("tiempo")
ax.set_ylabel("velocidad angular")

ax.set_xlim(0, Tfinal)
ax.set_ylim(-1.5, 1.5)

ax.grid()

ax.legend()


# ==========================================================
# TEXTO INFORMATIVO
# ==========================================================

texto_info = ax.text(
    0.02,
    0.95,
    "",
    transform=ax.transAxes,
    verticalalignment="top"
)


# ==========================================================
# SLIDERS DE MOMENTOS DE INERCIA
# ==========================================================

ax_I1 = plt.axes([0.20, 0.38, 0.65, 0.025])
ax_I2 = plt.axes([0.20, 0.34, 0.65, 0.025])
ax_I3 = plt.axes([0.20, 0.30, 0.65, 0.025])


slider_I1 = Slider(
    ax_I1,
    "I1",
    0.2,
    5.0,
    valinit=I1_inicial
)

slider_I2 = Slider(
    ax_I2,
    "I2",
    0.2,
    5.0,
    valinit=I2_inicial
)

slider_I3 = Slider(
    ax_I3,
    "I3",
    0.2,
    5.0,
    valinit=I3_inicial
)


# ==========================================================
# SLIDERS DE CONDICIONES INICIALES
# ==========================================================

ax_w1 = plt.axes([0.20, 0.22, 0.65, 0.025])
ax_w2 = plt.axes([0.20, 0.18, 0.65, 0.025])
ax_w3 = plt.axes([0.20, 0.14, 0.65, 0.025])


slider_w1 = Slider(
    ax_w1,
    "omega 1 inicial",
    -1.5,
    1.5,
    valinit=w1_inicial
)

slider_w2 = Slider(
    ax_w2,
    "omega 2 inicial",
    -1.5,
    1.5,
    valinit=w2_inicial
)

slider_w3 = Slider(
    ax_w3,
    "omega 3 inicial",
    -1.5,
    1.5,
    valinit=w3_inicial
)


# ==========================================================
# BOTÓN SIMULAR
# ==========================================================

ax_boton = plt.axes(
    [0.38, 0.05, 0.20, 0.05]
)

boton = Button(
    ax_boton,
    "SIMULAR"
)


# ==========================================================
# FUNCIÓN DEL BOTÓN
# ==========================================================

def actualizar(event):

    # Leer los valores actuales de los sliders

    I1 = slider_I1.val
    I2 = slider_I2.val
    I3 = slider_I3.val

    w10 = slider_w1.val
    w20 = slider_w2.val
    w30 = slider_w3.val


    # Nueva simulación

    t, U = simular(
        I1,
        I2,
        I3,
        w10,
        w20,
        w30
    )


    # Cambiar los datos de las curvas

    linea1.set_ydata(U[:, 0])
    linea2.set_ydata(U[:, 1])
    linea3.set_ydata(U[:, 2])


    # ------------------------------------------------------
    # Descubrir cuál es el eje intermedio
    # ------------------------------------------------------

    inercias = [
        (I1, "eje 1"),
        (I2, "eje 2"),
        (I3, "eje 3")
    ]

    inercias_ordenadas = sorted(inercias)

    eje_intermedio = inercias_ordenadas[1][1]


    # ------------------------------------------------------
    # Descubrir alrededor de qué eje estamos girando más
    # ------------------------------------------------------

    velocidades = [
        abs(w10),
        abs(w20),
        abs(w30)
    ]

    indice_dominante = velocidades.index(
        max(velocidades)
    )

    eje_dominante = indice_dominante + 1


    # Texto de diagnóstico

    mensaje = (
        f"Eje intermedio: {eje_intermedio}\n"
        f"Giro inicial principal: eje {eje_dominante}"
    )

    texto_info.set_text(mensaje)


    # Redibujar

    fig.canvas.draw_idle()


# El botón ejecuta actualizar()

boton.on_clicked(actualizar)


# ==========================================================
# PRIMER TEXTO
# ==========================================================

actualizar(None)


# ==========================================================
# MOSTRAR
# ==========================================================

plt.show()