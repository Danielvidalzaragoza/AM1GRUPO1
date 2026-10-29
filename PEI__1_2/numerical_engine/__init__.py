"""Public API for numerical integration, convergence, and stability."""

from .convergence import converge_sys, richardson_error
from .integrators import integrar_Cauchy, paso_Cauchy
from .methods import CN, EE, EI, LF, RK4
from .solvers import My_Jacobian, My_Newton
from .stability import (
    obtener_region_estabilidad,
    obtener_todas_regiones_estabilidad,
    stability_report,
)

__all__ = [
    "EE", "EI", "CN", "RK4", "LF", "My_Newton", "My_Jacobian",
    "paso_Cauchy", "integrar_Cauchy", "converge_sys", "richardson_error",
    "stability_report", "obtener_region_estabilidad",
    "obtener_todas_regiones_estabilidad",
]
