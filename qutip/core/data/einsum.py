import numpy as np
from .dense import Dense
from .dispatch import Dispatcher as _Dispatcher

__all__ = ['einsum', 'einsum_dense']


def einsum_dense(op0, /, *rest_operands, subscripts, tensor_shapes, out_shape):
    """Dense specialization for einsum."""
    from .convert import to as _to

    operands = (op0, *rest_operands)
    tensors = [
        _to(Dense, op).as_ndarray().reshape(shape)
        for op, shape in zip(operands, tensor_shapes)
    ]
    # TODO: when numpy is bumped to 2.2 add the optimize argument to np.einsum call
    result = np.einsum(subscripts, *tensors).reshape(out_shape)

    # With one operand, numpy may return a view of the input. Copy it so
    # the result does not share its memory.
    return Dense(result, copy=len(operands) == 1)


einsum = _Dispatcher(
    einsum_dense,
    name='einsum',
    inputs=('op0',),
    out=True
)
einsum.__doc__ = """
    Data layer implementation of Einstein summation.

    Parameters
    ----------
    op0 : Data
        The first operand, used for type dispatching.
    *rest_operands : tuple of Data
        The remaining QuTiP data objects to contract.
    subscripts : str
        The einsum equation, always with an output part (->). It labels
        the axes of each operand after the reshape to `tensor_shapes`.
    tensor_shapes : tuple of tuples
        The N-D tensor shapes for each operand.
    out_shape : tuple, optional
        The final 2D shape of the resulting Data object.
"""

einsum.add_specialisations([
    (Dense, Dense, einsum_dense),
], _defer=True)
