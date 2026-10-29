"""PEI 1 application entry point."""

import matplotlib.pyplot as plt

from GUI_actitud import PanelActitud


def main():
    plt.rcParams["toolbar"] = "None"
    panel = PanelActitud()
    gestor = panel.fig.canvas.manager
    if gestor is not None:
        gestor.set_window_title("PEI 1 · Dinámica de actitud")
        ventana = getattr(gestor, "window", None)
        for maximizar in (lambda: ventana.state("zoomed"),
                          lambda: ventana.showMaximized()):
            try:
                maximizar()
                break
            except Exception:
                pass
    plt.show()


if __name__ == "__main__":
    main()
