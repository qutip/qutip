"""
Einstein summation convention for Qobj.
"""

from collections import Counter
import numpy as np
from qutip.core.data import einsum as _data_einsum, extract
__all__ = ["einsum"]


def _infer_out_dims(subscripts, operands):
    """
    Infer the output dimensions for a quantum einsum operation.

    Parses the Einstein summation subscripts and the dimensions of the
    input operands to determine the row and column dimensions of the
    resulting quantum object.

    Only multiplication and contraction are supported.  If the subscripts
    would require an implicit transpose (contracting two row indices or
    two column indices across operands), a ``ValueError`` is raised so
    that the caller can apply ``.trans()`` / ``.dag()`` explicitly.

    Parameters
    ----------
    subscripts : str
        The Einstein summation string (e.g., 'ij,jk->ik').
    operands : sequence of Qobj
        The quantum objects being contracted.

    Returns
    -------
    list of list of int or None
        The output dimensions as ``[out_row_dims, out_col_dims]``. Returns
        ``None`` if all indices are contracted, resulting in a scalar.
    """
    from qutip.core.dimensions import flatten

    if "->" in subscripts:
        inputs_part, output_part = subscripts.split("->")
    else:
        inputs_part = subscripts
        all_labels = [c for c in inputs_part if c.isalnum()]
        counts = Counter(all_labels)
        output_part = "".join(sorted(
            [c for c, count in counts.items() if count == 1]
        ))

    input_subs = inputs_part.split(",")

    char_to_dim = {}
    char_occurrences = {}
    for op_idx, (op, sub) in enumerate(zip(operands, input_subs)):
        row_flat = flatten(op.dims[0])
        col_flat = flatten(op.dims[1])
        row_len = len(row_flat)
        for i, char in enumerate(sub):
            is_row = i < row_len
            char_to_dim[char] = (
                row_flat[i] if is_row else col_flat[i - row_len]
            )
            char_occurrences.setdefault(char, []).append((op_idx, is_row))

    # Identify contracted indices.
    contracted = {
        char: occ for char, occ in char_occurrences.items() if len(occ) > 1
    }

    # Reject implicit reordering.
    if len(operands) == 1 and not contracted:
        raise ValueError(
            "einsum only supports multiplication/contraction: subscripts "
            f"{subscripts!r} contract no repeated index for a single "
            "operand. Pure reordering or transpose is not supported; use "
            ".trans()/.dag()/.permute() instead."
        )

    # Reject row <-> row and col <-> col contraction.
    for char, occurrences in contracted.items():
        if len({is_row for _, is_row in occurrences}) != 2:
            raise ValueError(
                f"einsum subscript {char!r} in {subscripts!r} contracts a "
                "row with a row (or a col with a col), which would "
                "implicitly transpose an operand. Call .trans()/.dag() on "
                "that operand explicitly and rewrite the subscript instead."
            )

    # Reject implicit output transpose: all row-origin indices must appear
    # before all col-origin indices in the output.
    seen_col = False
    for char in output_part:
        _, is_row = char_occurrences[char][0]
        if is_row and seen_col:
            raise ValueError(
                f"einsum subscripts {subscripts!r} would implicitly transpose "
                f"the output: index {char!r} is a row-origin index but "
                "appears after a col-origin index in the output. Reorder the "
                "output subscripts so that all row indices precede all "
                "column indices, and apply .trans()/.dag() explicitly "
                "if a transpose is needed."
            )
        if not is_row:
            seen_col = True

    # Classify each surviving output character as row or column based
    # on its original role in the input operand where it first appeared.
    out_row_subs = []
    out_col_subs = []
    for char in output_part:
        _, is_row = char_occurrences[char][0]
        (out_row_subs if is_row else out_col_subs).append(char)

    if not output_part:
        return None

    out_row_dims = [char_to_dim[c] for c in out_row_subs] or [1]
    out_col_dims = [char_to_dim[c] for c in out_col_subs] or [1]
    return [out_row_dims, out_col_dims]


def einsum(subscripts, *operands, out_dims=None):
    """
    Implementation of numpy.einsum for Qobj.
    Evaluates the Einstein summation convention on the operands.

    Parameters
    ----------
    subscripts: str
        Specifies the subscripts for summation as comma
        separated list of subscript labels.
    operands: list of array_like
        These are the arrays for the operation.
    out_dims: list of list of int, optional
        The dimensions of the resulting quantum object.

    Returns
    -------
    Qobj (or numpy.complex128)
        Result of einsum as a Qobj, or a complex scalar if the operation
        contracts all indices.
    """
    from qutip.core.dimensions import dims_to_tensor_shape, dims_to_tensor_perm

    for op in operands:
        op._dims._require_pure_dims("einsum")

    data_operands = tuple(op.data for op in operands)

    if out_dims is None:
        out_dims = _infer_out_dims(subscripts, operands)

    tensor_shapes = tuple(
        dims_to_tensor_shape(op.dims)
        for op in operands
    )

    # tensor_perms and out_perm are required to map the physical matrix layout
    # of the operands (which might be permuted due to column-stacking
    # vectorization conventions for superoperators, operator-kets, or
    # operator-bras) to and from the logical subsystem
    # tensor layout used by einsum.
    tensor_perms = tuple(
        dims_to_tensor_perm(op.dims)
        for op in operands
    )

    if out_dims is not None:
        out_perm = dims_to_tensor_perm(out_dims)
        out_shape = (int(np.prod(out_dims[0])), int(np.prod(out_dims[1])))
    else:
        out_perm = (0,)
        # Enforce 1x1 shape for Cython dispatcher compatibility on scalars.
        out_shape = (1, 1)

    result_data = _data_einsum(
        *data_operands,
        subscripts=subscripts,
        tensor_shapes=tensor_shapes,
        tensor_perms=tensor_perms,
        out_perm=out_perm,
        out_shape=out_shape
    )

    # Extract scalar from the 1x1 Data object.
    if out_dims is None:
        return extract(result_data)[0, 0]

    # Get Qobj class from operand to avoid circular import
    Qobj_class = type(operands[0])
    return Qobj_class(result_data, dims=out_dims, copy=False)
