from ._base import Qobj, _QobjBuilder
from .. import data as _data
from typing import Any, Literal
import numpy as np


__all__ = []


class _StateQobj(Qobj):
    def proj(self) -> Qobj:
        """Form the projector from a given ket or bra vector.

        Parameters
        ----------
        Q : :class:`.Qobj`
            Input bra or ket vector

        Returns
        -------
        P : :class:`.Qobj`
            Projection operator.
        """
        dims = ([self._dims[0], self._dims[0]] if self.isket
                else [self._dims[1], self._dims[1]])
        return Qobj(_data.project(self._data),
                    dims=dims,
                    isherm=True,
                    copy=False)

    def norm(
        self,
        norm: Literal["l2", "max"] = "l2",
        kwargs: dict[str, Any] = None
    ) -> float:
        """
        Norm of a quantum object.

        Default norm is L2-norm.  Other
        ket and operator norms may be specified using the `norm` parameter.

        Parameters
        ----------
        norm : str, default: "l2"
            Which type of norm to use.  Allowed values are 'l2' and 'max'.

        kwargs : dict, optional
            Additional keyword arguments to pass on to the relevant norm
            solver.  See details for each norm function in :mod:`.data.norm`.

        Returns
        -------
        norm : float
            The requested norm of the operator or state quantum object.
        """
        norm = norm or "l2"
        if norm not in ["l2", "max"]:
            raise ValueError(
                "vector norm must be in {'k2', 'max'}"
            )

        kwargs = kwargs or {}
        return {
            'l2': _data.norm.l2,
            'max': _data.norm.max
        }[norm](self._data, **kwargs)

    def unit(
        self,
        inplace: bool = False,
        norm: Literal["l2", "max"] = "l2",
        kwargs: dict[str, Any] = None
    ) -> Qobj:
        """
        Operator or state normalized to unity.  Uses norm from Qobj.norm().

        Parameters
        ----------
        inplace : bool, default: False
            Do an in-place normalization
        norm : str, default: "l2"
            Requested norm for states / operators.
        kwargs : dict, optional
            Additional key-word arguments to be passed on to the relevant norm
            function (see :meth:`.norm` for more details).

        Returns
        -------
        obj : :class:`.Qobj`
            Normalized quantum object.  Will be the `self` object if in place.
        """
        norm_ = self.norm(norm=norm, kwargs=kwargs)
        if inplace:
            self.data = _data.mul(self.data, 1 / norm_)
            out = self
        else:
            out = self / norm_
        return out

    def basis_expansion(self, decimal_places: int = 14,
                        term_separator: str = " + ",
                        dim_separator: str = "auto",
                        sort: str = "largest-first") -> str:
        """
        Return a string-representation of the basis expansion
        for a ket or a bra, e.g.

        ``"(0.5+0.15j) |010> + (0.25+0j) |000> + ..."``

        Parameters
        ----------
        decimal_places : int
            The number of decimal places to show for the coefficients.

        term_separator: str
            Separator string between terms. E.g. use " +\n"
            to print terms on separate lines.

        dim_separator: str
            Separator string between dimension indices.
            E.g. use ", " to print as ``"|0, 0, 0>"``.
            If dim_separator="auto", then "" is used
            for qubits and ", " is used otherwise.

        sort: str
            Show largest elements (by magnitude) first
            if "largest-first". Unsorted otherwise.

        Returns
        -------
        basis_expansion : str
            The requested string representation.
        """

        if not (self.isket or self.isbra):
            raise ValueError(
                f"Only implemented for states, not {self.type}."
            )
        dims = self.dims[0 if self.isket else 1]

        if dim_separator == "auto":
            # No separator for pure qubit states but comma-separator otherwise,
            # since bitstrings are nice, but e.g. |153> would be ambiguous.
            dim_separator = ", " if any([dim > 2 for dim in dims]) else ""

        template = "{} |{}>" if self.isket else "{} <{}|"

        ket_full = np.round(self.full(squeeze=True), decimals=decimal_places)

        indices = range(len(ket_full))
        if sort == "largest-first":
            indices = np.argsort(np.abs(ket_full))[::-1]

        parts = []
        for i in indices:
            if np.abs(ket_full[i]) > 10**-(decimal_places):
                coeff = ket_full.item(i)
                basis_str = dim_separator.join(
                    map(str, np.unravel_index(i, dims))
                )
                parts.append(template.format(coeff, basis_str))

        if len(parts) == 0:  # return something for norm-zero states.
            return "0"

        return term_separator.join(parts)


class Bra(_StateQobj):
    def __init__(self, data, dims, **flags):
        super().__init__(data, dims, **flags)
        if self._dims.type != "bra":
            raise ValueError(
                f"Expected bra dimensions, but got {self._dims.type}"
            )

    @property
    def isbra(self) -> bool:
        return True


class Ket(_StateQobj):
    def __init__(self, data, dims, **flags):
        super().__init__(data, dims, **flags)
        if self._dims.type != "ket":
            raise ValueError(
                f"Expected ket dimensions, but got {self._dims.type}"
            )

    @property
    def isket(self) -> bool:
        return True


class OperKet(_StateQobj):
    def __init__(self, data, dims, **flags):
        super().__init__(data, dims, **flags)
        if self._dims.type != "operator-ket":
            raise ValueError(
                f"Expected ket dimensions, but got {self._dims.type}"
            )

    @property
    def isoperket(self) -> bool:
        return True


class OperBra(_StateQobj):
    def __init__(self, data, dims, **flags):
        super().__init__(data, dims, **flags)
        if self._dims.type != "operator-bra":
            raise ValueError(
                f"Expected ket dimensions, but got {self._dims.type}"
            )

    @property
    def isoperbra(self) -> bool:
        return True


_QobjBuilder.qobjtype_to_class["ket"] = Ket
_QobjBuilder.qobjtype_to_class["bra"] = Bra
_QobjBuilder.qobjtype_to_class["operator-ket"] = OperKet
_QobjBuilder.qobjtype_to_class["operator-bra"] = OperBra
