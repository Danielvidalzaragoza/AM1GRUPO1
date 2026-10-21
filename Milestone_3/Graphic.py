import matplotlib.pyplot as plt

def grafico_sol(soluciones: dict):
    fig, ax = plt.subplots()
    for esquema, solucion in soluciones.items():
        ax.plot(solucion[:, 0], solucion[:, 1], label=esquema)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.scatter([1, -1], [0, 0], color="black", zorder=5)
    ax.annotate("(1, 0)", (1, 0), xytext=(6, 6), textcoords="offset points")
    ax.annotate("(-1, 0)", (-1, 0), xytext=(6, 6), textcoords="offset points")
    ax.axis("equal")
    ax.set_xlabel("Posición x")
    ax.set_ylabel("Velocidad v")
    ax.set_title("Órbita del oscilador armónico simple")
    ax.legend()
    return fig

def grafico_error(tiempos: list, errores: dict):
    fig, ax = plt.subplots()
    for esquema, error in errores.items():
        ax.plot(tiempos, error[:, 0], label=f"Error en x ({esquema})")
        ax.plot(tiempos, error[:, 1], label=f"Error en v ({esquema})")
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Error")
    ax.set_title("Error temporal estimado mediante Richardson")
    ax.legend()
    return fig
    