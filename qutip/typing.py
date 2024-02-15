from typing import Sequence, Union, Any, Callable

# from .core.cy.qobjEvo import QobjEvoLike, Element
# from .core.coeffients import CoefficientLike
from numbers import Number
from typing_extensions import Protocol
import numpy as np
import scipy.interpolate


__all__ = ["QobjEvoLike", "CoefficientLike", "LayerType"]


class QEvoFunction(Protocol):
    def __call__(self, t: Number, **kwargs) -> "Qobj":
        ...


CoefficientLike = Union[
    "Coefficient",
    str,
    Callable[[float, ...], complex],
    np.ndarray,
    scipy.interpolate.PPoly,
    scipy.interpolate.BSpline,
    Any,
]

ElementType = Union[QEvoFunction, "Qobj", tuple["Qobj", CoefficientLike]]

QobjEvoLike = Union["Qobj", "QobjEvo", ElementType, Sequence[ElementType]]

LayerType = Union[str, type]
