from collections import Counter
from functools import lru_cache, partial
from itertools import chain
from math import prod

import numpy as np
from .dense import Dense
from .dispatch import Dispatcher as _Dispatcher

__all__ = ['einsum', 'einsum_dense']


def _is_pairwise(terms, output):
    """
    Whether numpy.tensordot can do the contraction, each label must be summed
    over two operands or kept in the output from one operand.
    """
    strings = (*terms, output)
    return (
        len(terms) > 1
        and all(n == 2 for n in Counter(chain(*strings)).values())  # chain(*strings) does a character count
        and all(len(set(labels)) == len(labels) for labels in strings)
    )


@lru_cache(maxsize=1024)
def _contraction(subscripts, tensor_shapes):
    inputs, _, output = subscripts.partition("->")
    terms = inputs.split(",")
    size = dict(zip(chain(*terms), chain(*tensor_shapes)))

    if not _is_pairwise(terms, output) or prod(size.values()) < 2**12:
        # In my benchmarks, np.einsum was faster for < 12 qubits
        # But np.tensordot faster for >= 12 qubits
        # This overhead in np.tensordot was due to internal Python calls in np.tensordot
        return partial(np.einsum, subscripts)

    shapes = [np.broadcast_to(0, shape) for shape in tensor_shapes]
    path = np.einsum_path(subscripts, *shapes, optimize="greedy")[0][1:]

    if any(len(pair) != 2 for pair in path):
        return partial(np.einsum, subscripts)

    steps, labels = _tensordot_steps(terms, size, path)
    return partial(_tensordot, steps, [labels.index(c) for c in output])


def _tensordot_steps(terms, size, path):
    """
    Turn a path from numpy.einsum_path into a list of tensordot calls.
    Step (i, j, axes) contracts tensors i and j and adds the result at the end of the tensors.

    Return the steps and the labels of the last result.
    """
    # Like numpy take each pair out of the list and add the result at the end
    operands = list(enumerate(terms))
    steps = []
    for pair in path:
        (i, x), (j, y) = sorted(
            [operands.pop(k) for k in sorted(pair, reverse=True)],
            key=lambda operand: prod(map(size.get, operand[1])),
        )
        shared = [label for label in x if label in y]
        axes = ([x.index(c) for c in shared], [y.index(c) for c in shared])
        steps.append((i, j, axes))

        labels = "".join(label for label in x + y if label not in shared)
        operands.append((len(terms) + len(steps) - 1, labels))

    [(_, labels)] = operands
    return steps, labels


def _tensordot(steps, out_axes, *tensors):
    # Contract the tensors with the steps (from _tensordot_steps)
    tensors = list(tensors)
    for i, j, axes in steps:
        tensors.append(np.tensordot(tensors[i], tensors[j], axes))
    return tensors[-1].transpose(out_axes)


def einsum_dense(op0, /, *rest_operands, subscripts, tensor_shapes, out_shape):
    """Dense specialization for einsum."""
    from .convert import to as _to

    operands = (op0, *rest_operands)
    tensors = [
        _to(Dense, op).as_ndarray().reshape(shape)
        for op, shape in zip(operands, tensor_shapes)
    ]
    contract = _contraction(subscripts, tensor_shapes)
    result = contract(*tensors).reshape(out_shape)

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
