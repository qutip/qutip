"""
Einstein summation convention for Qobj.
"""

from collections import Counter
from functools import lru_cache, partial

from qutip.core.data import einsum as _data_einsum, extract
from .dimensions import Dimensions, dims_to_tensor_perm, dims_to_tensor_shape


__all__ = ["einsum"]


def _to_data_axes(labels, dims):
    """Reorder labels from the order of dims to the data axes."""

    # The two orders differ only for superoperators and operator-kets:
    # their data stacks the columns of an operator.
    axes = dims_to_tensor_perm(dims)
    if len(labels) != len(axes):
        raise ValueError(
            f"einsum subscripts {labels!r} need one label per entry of "
            f"dims {dims}."
        )
    return "".join(label for _, label in sorted(zip(axes, labels)))


def _infer_out_dims(inputs, output, op_dims):
    """
    Return the dims of the output, or None for a scalar.

    Raise ValueError if the subscripts would transpose an operand or the
    output. The user must ask for that with ``.trans()`` or ``.dag()``.
    """
    size = {}
    is_row = {}  # label -> one bool per use: is it a row index there?
    for labels, dims in zip(inputs, op_dims):
        rows, cols = dims.flat()
        for i, (label, n) in enumerate(zip(labels, rows + cols)):
            size[label] = n
            is_row.setdefault(label, []).append(i < len(rows))

    contracted = {
        label: uses for label, uses in is_row.items() if len(uses) > 1
    }
    if len(op_dims) == 1 and not contracted:
        raise ValueError(
            f"einsum on one operand must contract an index, but "
            f"{inputs[0]!r} repeats no label. To reorder indices, use "
            ".trans(), .dag() or .permute()."
        )
    for label, uses in contracted.items():
        if len(set(uses)) == 1:
            raise ValueError(
                f"einsum label {label!r} joins two row indices or two "
                "column indices. Apply .trans() or .dag() to one operand "
                "first."
            )

    # An output index is a row index if its label was first used on a row.
    out_is_row = [is_row[label][0] for label in output]

    # Sorted in reverse, True (a row) comes before False (a column).
    if out_is_row != sorted(out_is_row, reverse=True):
        raise ValueError(
            f"einsum output {output!r} has a column index before a row "
            "index. Put the row indices first, then apply .trans() or "
            ".dag() to the result if needed."
        )

    if not output:
        return None

    n_rows = sum(out_is_row)
    if not 0 < n_rows < len(output):
        raise ValueError(
            f"einsum output {output!r} needs at least one row index and "
            "one column index, or no index at all for a scalar."
        )

    sizes = [size[label] for label in output]
    return Dimensions([sizes[:n_rows], sizes[n_rows:]])


@lru_cache(maxsize=1024)
def _contraction(subscripts, op_dims, out_dims):
    """
    Return the data layer function that computes the einsum, and the dims
    of its output (None for a scalar).

    Nothing here depends on the data, so the work is done once for each
    set of subscripts and dims and then reused.
    """
    for dims in op_dims:
        dims._require_pure_dims("einsum")

    if "->" in subscripts:
        inputs, output = subscripts.split("->")
    else:
        # Same default as numpy (the labels used once sorted)
        inputs = subscripts
        counts = Counter(filter(str.isalnum, inputs))
        output = "".join(sorted(c for c, n in counts.items() if n == 1))

    inputs = inputs.split(",")

    if len(inputs) != len(op_dims):
        raise ValueError(
            f"einsum subscripts {subscripts!r} describe {len(inputs)} "
            f"operands, but {len(op_dims)} were given."
        )

    data_inputs = ",".join(map(_to_data_axes, inputs, op_dims))
    inferred = _infer_out_dims(inputs, output, op_dims)
    if out_dims is None:
        out_dims = inferred

    # out_dims may only group the output indices in another way, for
    # example as an operator-ket. It cannot change their sizes or order.
    elif inferred is None or out_dims.flat() != inferred.flat():
        raise ValueError(
            f"out_dims {out_dims} does not match the einsum output "
            f"{output!r}, which has dims {inferred}."
        )

    if out_dims is None:
        # Data is always 2D, so a scalar comes back as a 1x1 matrix.
        data_output, out_shape = "", (1, 1)
    else:
        data_output = _to_data_axes(output, out_dims)
        out_shape = out_dims.shape
    contract = partial(
        _data_einsum,
        subscripts=f"{data_inputs}->{data_output}",
        tensor_shapes=tuple(map(dims_to_tensor_shape, op_dims)),
        out_shape=out_shape,
    )
    return contract, out_dims


def einsum(subscripts, *operands, out_dims=None):
    """
    Einstein summation for Qobj, like :func:`numpy.einsum`.

    Parameters
    ----------
    subscripts: str
        Labels of each operand, separated by commas, then optionally
        -> and the labels of the output. An operand needs one label
        per entry of its row dims, then one per entry of its column dims.
    operands: Qobj
        The objects to contract.
    out_dims: list of list of int, optional
        Dims of the output, to group its indices as an operator-ket or a
        superoperator. By default the output has one flat list of row dims
        and one of column dims.

    Returns
    -------
    Qobj (or numpy.complex128)
        Result of einsum as a Qobj, or a complex scalar if the operation
        contracts all indices.
    """
    contract, out_dims = _contraction(
        subscripts,
        tuple(op._dims for op in operands),
        None if out_dims is None else Dimensions(out_dims),
    )

    result_data = contract(*(op.data for op in operands))

    # Extract scalar from the 1x1 Data object.
    if out_dims is None:
        return extract(result_data)[0, 0]

    # Get Qobj class from operand to avoid circular import
    Qobj_class = type(operands[0])
    return Qobj_class(result_data, dims=out_dims, copy=False)
