
from .qobj import Qobj
from qutip.typing import LayerType
from . import data as _data
import qutip
from .dimensions import flatten
import numbers


__all__ = ["ptrace"]


def ptrace(Q: Qobj, sel: int | list[int], dtype: LayerType = None) -> Qobj:
    """
    Partial trace of the Qobj with selected components remaining.

    Parameters
    ----------
    Q : :class:`.Qobj`
        Composite quantum object.
    sel : int/list
        An ``int`` or ``list`` of components to keep after partial trace.

    Returns
    -------
    oper : :class:`.Qobj`
        Quantum object representing partial trace with selected components
        remaining.

    dtype : type, str
        The matrix format of output.

    Notes
    -----
    This function is for legacy compatibility only. It is recommended to use
    the ``ptrace()`` Qobj method.
    """
    if not isinstance(Q, Qobj):
        raise TypeError("Input is not a quantum object")

    try:
        sel = sorted(sel)
    except TypeError:
        if not isinstance(sel, numbers.Integral):
            raise TypeError(
                "selection must be an integer or list of integers"
            ) from None
        sel = [sel]
    if Q.isoperket:
        dims = Q.dims[0]
        data = qutip.vector_to_operator(Q).data
    elif Q.isoperbra:
        dims = Q.dims[1]
        data = qutip.vector_to_operator(Q.dag()).data
    elif Q.issuper or Q.isoper:
        dims = Q.dims
        data = Q.data
    else:
        dims = [Q.dims[0] if Q.isket else Q.dims[1]] * 2
        data = _data.project(Q.data)
    if dims[0] != dims[1]:
        raise ValueError("partial trace is not defined on non-square maps")
    dims = flatten(dims[0])
    new_data = _data.ptrace(data, dims, sel, dtype=dtype)
    new_dims = [[dims[x] for x in sel]] * 2 if sel else None
    out = Qobj(new_data, dims=new_dims, copy=False)
    if Q.isoperket:
        return qutip.operator_to_vector(out)
    if Q.isoperbra:
        return qutip.operator_to_vector(out).dag()
    return out
