from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from numerical_engine import LF, RK4, integrar_Cauchy


OUTPUT_DIR = Path(__file__).resolve().parent
H = 0.6
T_FINAL = 1.2
U0 = np.array([1.0])


def f(u: np.ndarray, _t: float) -> np.ndarray:
    return u


def jacobian(_u: np.ndarray, _t: float) -> np.ndarray:
    return np.eye(1)


def dibujar_pendiente(
    ax: plt.Axes,
    t_eval: float,
    y_eval: float,
    pendiente: float,
    etiqueta: str | None = None,
    anotacion: str | None = None,
) -> None:
    longitud = H * 0.18
    ax.plot(
        [t_eval, t_eval + longitud],
        [y_eval, y_eval + pendiente * longitud],
        color="green",
        linestyle="-.",
        lw=2,
        label=etiqueta,
    )
    ax.plot(t_eval, y_eval, marker="o", color="green", ms=4)
    if anotacion:
        ax.annotate(
            anotacion,
            (t_eval, y_eval),
            xytext=(3, 5),
            textcoords="offset points",
            fontsize=9,
            color="darkgreen",
        )


def crear_grafica(esquema: str, titulo: str) -> None:
    if esquema == "LF":
        t_n = 0.0
        u_n = U0
        t_prev = t_n - H
        u_prev = RK4(u_n, -H, t_n, f)
        t_final = t_n + H
        u_final = LF(u_n, H, t_n, f, u_prev)
        t = np.array([t_prev, t_n, t_final])
        u = np.vstack((u_prev, u_n, u_final))
    else:
        t, u = integrar_Cauchy(
            esquema,
            U0,
            1,
            0.0,
            H,
            f,
            jacobian,
        )
        t_final = t[-1]

    y_final = u[-1, 0]
    y_exacto = np.exp(t_final)
    curva_t = np.linspace(t[0], T_FINAL, 300)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(curva_t, np.exp(curva_t), "b", lw=2, label="Solución exacta")
    ax.plot(t, u[:, 0], "r--", lw=2, label=f"Método {esquema}")
    ax.plot(t[0], u[0, 0], "ko", ms=8, label="Condición inicial")

    if esquema == "LF":
        ax.plot(
            t[0],
            u[0, 0],
            "o",
            color="darkorange",
            ms=7,
            label=r"$N-1$ calculado con RK4",
        )

    ax.plot(t_final, y_final, "ro", ms=8, label=r"$y_{N+1}$ numérico")
    ax.plot(t_final, y_exacto, "bo", ms=8, label=r"$y(t_{N+1})$ exacto")
    ax.plot([t_final, t_final], [y_final, y_exacto], "k:", lw=1.5)
    if esquema == "EI":
        dibujar_pendiente(
            ax,
            t[-1],
            y_exacto,
            f(np.array([y_exacto]), t[-1])[0],
            etiqueta=r"Tangente de la solución exacta en $N+1$",
        )
    elif esquema == "RK4":
        k1 = f(u[0], t[0])
        estado_k2 = u[0] + H * k1 / 2
        k2 = f(estado_k2, t[0] + H / 2)
        estado_k3 = u[0] + H * k2 / 2
        k3 = f(estado_k3, t[0] + H / 2)
        estado_k4 = u[0] + H * k3
        k4 = f(estado_k4, t[0] + H)
        etapas = [
            (t[0], u[0, 0], k1[0], r"$k_1$"),
            (t[0] + H / 2, estado_k2[0], k2[0], r"$k_2$"),
            (t[0] + H / 2, estado_k3[0], k3[0], r"$k_3$"),
            (t[0] + H, estado_k4[0], k4[0], r"$k_4$"),
        ]
        for indice, (t_eval, y_eval, pendiente, nombre) in enumerate(etapas):
            dibujar_pendiente(
                ax,
                t_eval,
                y_eval,
                pendiente,
                etiqueta="Pendientes de las etapas RK4" if indice == 0 else None,
                anotacion=nombre,
            )
    elif esquema == "CN":
        dibujar_pendiente(
            ax,
            t[0],
            u[0, 0],
            f(u[0], t[0])[0],
            etiqueta=r"Pendientes promediadas por CN",
        )
        dibujar_pendiente(
            ax,
            t[-1],
            u[-1, 0],
            f(u[-1], t[-1])[0],
        )
    else:
        dibujar_pendiente(
            ax,
            t[1],
            u[1, 0],
            f(u[1], t[1])[0],
            etiqueta=r"Pendiente evaluada en $N$",
        )
    ax.text(
        t_final + 0.02,
        (y_final + y_exacto) / 2,
        "error local",
    )

    if esquema == "LF":
        ax.set_xticks(t, ["N-1", "N", "N+1"])
        ax.plot(t[1], u[1, 0], "ko", ms=8)
    else:
        ax.set_xticks([t[0], t_final], ["N", "N+1"])
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.set_title(titulo)
    ax.set_xlim(left=t[0] - 0.15, right=T_FINAL + 0.35)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"grafica-{esquema}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    graficas = {
        "EI": "Método de Euler implícito (EI)",
        "RK4": "Método de Runge-Kutta de orden 4 (RK4)",
        "CN": "Método de Crank-Nicolson (CN)",
        "LF": "Método Leapfrog (LF)",
    }
    for esquema, titulo in graficas.items():
        crear_grafica(esquema, titulo)
