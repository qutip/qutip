from typing import Sequence, Union, Any, Protocol
from numbers import Number, Real
import numpy as np
import scipy.interpolate


__all__ = ["QobjEvoLike", "CoefficientLike", "LayerType"]


class QEvoProtocol(Protocol):
    def __call__(self, t: Real, **kwargs) -> "Qobj":
        ...


class CoeffProtocol(Protocol):
    def __call__(self, t: Real, **kwargs) -> Number:
        ...


CoefficientLike = Union[
    "Coefficient",
    str,
    CoeffProtocol,
    np.ndarray,
    scipy.interpolate.PPoly,
    scipy.interpolate.BSpline,
    Any,
]


ElementType = Union[QEvoProtocol, "Qobj", tuple["Qobj", CoefficientLike]]


QobjEvoLike = Union["Qobj", "QobjEvo", ElementType, Sequence[ElementType]]


LayerType = Union[str, type]


SpaceLike = Union[int, list[int], list[list[int]], "Space"]


DimensionLike = Union[
    list[list[int], list[int]],
    list[list[list[int]], list[list[int]]],
    list["Space", "Space"],
    "Dimensions",
]
