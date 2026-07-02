import numpy as np
from .dense import Dense
from .dispatch import Dispatcher as _Dispatcher

__all__ = ['einsum', 'einsum_dense']


def einsum_dense(
        op0, /,
        subscripts,
        rest_operands,
        tensor_shapes,
        out_shape=None
):
    """
    Dense CPU specialization for einsum.
    """
    operands = (op0,) + tuple(rest_operands)

    tensors = [
        op.as_ndarray().reshape(shape)
        for op, shape in zip(operands, tensor_shapes)
    ]

    result = np.einsum(subscripts, *tensors)

    # Scalar result
    if result.shape == ():
        return complex(result)

    if out_shape is None:
        half = result.ndim // 2
        rows = int(np.prod(result.shape[:half]))
        cols = int(np.prod(result.shape[half:]))
        out_shape = (rows, cols)

    return Dense(result.reshape(out_shape))


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
    subscripts : str
        The einsum subscript equation.
    rest_operands : tuple of Data
        The remaining QuTiP data objects to contract.
    tensor_shapes : tuple of tuples
        The N-D tensor shapes for each operand.
    out_shape : tuple, optional
        The final 2D shape of the resulting Data object.
"""

einsum.add_specialisations([
    (Dense, Dense, einsum_dense),
], _defer=True)
