import matplotlib.pyplot as plt

FONDO = "#0a0f1e"

TARJETA = "#111a2e"

BORDE = "#25314d"

CONTROL = "#1b2640"

CONTROL_HOVER = "#26345a"

TEXTO = "#e8ecf6"

TENUE = "#8b95ad"

ACENTO = "#4f8cff"

COLOR_EJES = ("#ff6b6b", "#ffd166", "#4cc9f0")

COLOR_L = "#c792ea"

COLOR_W = "#80ffb4"

COLOR_T = "#5b8def"

BIEN = "#80ffb4"

MAL = "#ff6b6b"

AVISO = "#ffb347"

MONO = ["Consolas", "DejaVu Sans Mono"]

ESQUEMAS = {
    "EE    Euler explícito": "EE",
    "EI    Euler implícito": "EI",
    "CN    Crank–Nicolson": "CN",
    "RK4  Runge–Kutta 4": "RK4",
    "LF    Leapfrog": "LF",
}

EJES = ("e₁   I₁ mínimo", "e₂   I₂ intermedio", "e₃   I₃ máximo")

SUB = "₁₂₃"

ANCHO, ALTO = 16.5, 8.4

TODAS = (0, 1, 2)

ESTADOS = {
    "listo": ("●  LISTO", TENUE),
    "marcha": ("●  EN CURSO", BIEN),
    "pausa": ("●  EN PAUSA", AVISO),
    "completado": ("●  COMPLETADO", ACENTO),
    "diverge": ("●  EL ESQUEMA DIVERGE", MAL),
}


def _aplicar_estilo():
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "text.color": TEXTO,
        "axes.labelcolor": TENUE,
        "axes.edgecolor": BORDE,
        "xtick.color": TENUE,
        "ytick.color": TENUE,
        "grid.color": BORDE,
        "grid.linewidth": 0.6,
        "figure.facecolor": FONDO,
        "axes.facecolor": TARJETA,
        "text.hinting": "no_hinting",
    })
