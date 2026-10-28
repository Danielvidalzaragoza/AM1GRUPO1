import matplotlib.pyplot as plt
import numpy as np

def grafico_sol(soluciones: dict):
    fig, ax = plt.subplots()
    for esquema, solucion in soluciones.items():
        ax.plot(solucion[:, 0], solucion[:, 1], label=esquema)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.scatter([1, -1], [0, 0], color="black", zorder=5)
    ax.annotate("(1, 0)", (1, 0), xytext=(6, 6), textcoords="offset points")
    ax.annotate("(-1, 0)", (-1, 0), xytext=(6, 6), textcoords="offset points")
    ax.set_aspect("equal", adjustable="box")
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

def grafico_convergencia(error_final: dict, pasos: dict):
    fig, ax = plt.subplots()
    for esquema, error in error_final.items():
        ax.plot(pasos[esquema], error, marker="o", label=esquema)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Número de pasos de tiempo")
    ax.set_ylabel(r"Diferencia final $\|U_N-U_{2N}\|$")
    ax.set_title("Convergencia del sistema")
    ax.legend()
    return fig