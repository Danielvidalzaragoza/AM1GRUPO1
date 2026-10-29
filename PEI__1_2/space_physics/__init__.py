"""Public API for the free-rotation physics model."""

from .attitude import actitud, paso_actitud
from .dynamics import ecuaciones_rotacion, jacobiano
from .equilibria import estabilidad_ejes, separatriz
from .invariants import historial_invariantes, invariantes
from .stability import estabilidad

__all__ = [
    "ecuaciones_rotacion", "jacobiano", "invariantes", "historial_invariantes",
    "estabilidad_ejes", "separatriz", "paso_actitud", "actitud", "estabilidad",
]
