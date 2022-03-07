# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################
"""
Internal use module for manipulating dims specifications.
"""

# Everything should be explicitly imported, not made available by default.
__all__ = []

import numpy as np
from operator import getitem
from functools import partial


def is_scalar(dims):
    """
    Returns True if a dims specification is effectively
    a scalar (has dimension 1).
    """
    return np.prod(flatten(dims)) == 1


def is_vector(dims):
    return (
        isinstance(dims, list) and
        isinstance(dims[0], (int, np.integer))
    )


def is_vectorized_oper(dims):
    return (
        isinstance(dims, list) and
        isinstance(dims[0], list)
    )


def type_from_dims(dims, enforce_square=False):
    bra_like, ket_like = map(is_scalar, dims)

    if bra_like:
        if is_vector(dims[1]):
            return 'bra'
        elif is_vectorized_oper(dims[1]):
            return 'operator-bra'

    if ket_like:
        if is_vector(dims[0]):
            return 'ket'
        elif is_vectorized_oper(dims[0]):
            return 'operator-ket'

    elif is_vector(dims[0]) and (dims[0] == dims[1] or not enforce_square):
        return 'oper'

    elif (
        is_vectorized_oper(dims[0])
        and (
            not enforce_square
            or (dims[0] == dims[1] and dims[0][0] == dims[1][0])
        )
    ):
        return 'super'
    return 'other'


def flatten(l):
    """Flattens a list of lists to the first level.

    Given a list containing a mix of scalars and lists,
    flattens down to a list of the scalars within the original
    list.

    Examples
    --------

    >>> flatten([[[0], 1], 2]) # doctest: +SKIP
    [0, 1, 2]

    """
    if not isinstance(l, list):
        return [l]
    else:
        return sum(map(flatten, l), [])


def deep_remove(l, *what):
    """Removes scalars from all levels of a nested list.

    Given a list containing a mix of scalars and lists,
    returns a list of the same structure, but where one or
    more scalars have been removed.

    Examples
    --------

    >>> deep_remove([[[[0, 1, 2]], [3, 4], [5], [6, 7]]], 0, 5) # doctest: +SKIP
    [[[[1, 2]], [3, 4], [], [6, 7]]]

    """
    if isinstance(l, list):
        # Make a shallow copy at this level.
        l = l[:]
        for to_remove in what:
            if to_remove in l:
                l.remove(to_remove)
            else:
                l = [deep_remove(elem, to_remove) for elem in l]
    return l


def unflatten(l, idxs):
    """Unflattens a list by a given structure.

    Given a list of scalars and a deep list of indices
    as produced by `flatten`, returns an "unflattened"
    form of the list. This perfectly inverts `flatten`.

    Examples
    --------

    >>> l = [[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]] # doctest: +SKIP
    >>> idxs = enumerate_flat(l) # doctest: +SKIP
    >>> unflatten(flatten(l), idxs) == l # doctest: +SKIP
    True

    """
    acc = []
    for idx in idxs:
        if isinstance(idx, list):
            acc.append(unflatten(l, idx))
        else:
            acc.append(l[idx])
    return acc


def _enumerate_flat(l, idx=0):
    if not isinstance(l, list):
        # Found a scalar, so return and increment.
        return idx, idx + 1
    else:
        # Found a list, so append all the scalars
        # from it and recurse to keep the increment
        # correct.
        acc = []
        for elem in l:
            labels, idx = _enumerate_flat(elem, idx)
            acc.append(labels)
        return acc, idx


def _collapse_composite_index(dims):
    """
    Given the dimensions specification for a composite index
    (e.g.: [2, 3] for the right index of a ket with dims [[1], [2, 3]]),
    returns a dimensions specification for an index of the same shape,
    but collapsed to a single "leg." In the previous example, [2, 3]
    would collapse to [6].
    """
    return [np.prod(dims)]


def _collapse_dims_to_level(dims, level=1):
    """
    Recursively collapses all indices in a dimensions specification
    appearing at a given level, such that the returned dimensions
    specification does not represent any composite systems.
    """
    if level == 0:
        return _collapse_composite_index(dims)
    return [_collapse_dims_to_level(index, level=level - 1) for index in dims]


def collapse_dims_oper(dims):
    """
    Given the dimensions specifications for a ket-, bra- or oper-type
    Qobj, returns a dimensions specification describing the same shape
    by collapsing all composite systems. For instance, the bra-type
    dimensions specification ``[[2, 3], [1]]`` collapses to
    ``[[6], [1]]``.

    Parameters
    ----------

    dims : list of lists of ints
        Dimensions specifications to be collapsed.

    Returns
    -------

    collapsed_dims : list of lists of ints
        Collapsed dimensions specification describing the same shape
        such that ``len(collapsed_dims[0]) == len(collapsed_dims[1]) == 1``.
    """
    return _collapse_dims_to_level(dims, 1)


def collapse_dims_super(dims):
    """
    Given the dimensions specifications for an operator-ket-, operator-bra- or
    super-type Qobj, returns a dimensions specification describing the same shape
    by collapsing all composite systems. For instance, the super-type
    dimensions specification ``[[[2, 3], [2, 3]], [[2, 3], [2, 3]]]`` collapses to
    ``[[[6], [6]], [[6], [6]]]``.

    Parameters
    ----------

    dims : list of lists of ints
        Dimensions specifications to be collapsed.

    Returns
    -------

    collapsed_dims : list of lists of ints
        Collapsed dimensions specification describing the same shape
        such that ``len(collapsed_dims[i][j]) == 1`` for ``i`` and ``j``
        in ``range(2)``.
    """
    return _collapse_dims_to_level(dims, 2)


def enumerate_flat(l):
    """Labels the indices at which scalars occur in a flattened list.

    Given a list containing a mix of scalars and lists,
    returns a list of the same structure, where each scalar
    has been replaced by an index into the flattened list.

    Examples
    --------

    >>> print(enumerate_flat([[[10], [20, 30]], 40])) # doctest: +SKIP
    [[[0], [1, 2]], 3]

    """
    return _enumerate_flat(l)[0]


def deep_map(fn, collection, over=(tuple, list)):
    if isinstance(collection, over):
        return type(collection)(deep_map(fn, el, over) for el in collection)
    else:
        return fn(collection)


def dims_to_tensor_perm(dims):
    """
    Given the dims of a Qobj instance, returns a list representing
    a permutation from the flattening of that dims specification to
    the corresponding tensor indices.

    Parameters
    ----------

    dims : list
        Dimensions specification for a Qobj.

    Returns
    -------

    perm : list
        A list such that ``data[flatten(dims)[idx]]`` gives the
        index of the tensor ``data`` corresponding to the ``idx``th
        dimension of ``dims``.
    """
    # We figure out the type of the dims specification,
    # relaxing the requirement that operators be square.
    # This means that dims_type need not coincide with
    # Qobj.type, but that works fine for our purposes here.
    dims_type = type_from_dims(dims, enforce_square=False)
    perm = enumerate_flat(dims)
    if dims_type in ('oper', 'ket', 'bra'):
        return flatten(perm)

    # If the type is other, we need to figure out if the
    # dims is superlike on its outputs and inputs
    # This is the case if the dims type for left or right
    # are, respectively, oper-like.
    if dims_type == 'other':
        raise NotImplementedError("Not yet implemented for type='other'.")

    # If we're still here, the story is more complicated. We'll
    # follow the strategy of creating a permutation by using
    # enumerate_flat then transforming the result to swap
    # input and output indices of vectorized matrices, then flattening
    # the result. We'll then rebuild indices using this permutation.
    if dims_type in ('operator-ket', 'super'):
        # Swap the input and output spaces of the right part of
        # perm.
        perm[1] = list(reversed(perm[1]))
    if dims_type in ('operator-bra', 'super'):
        # Ditto, but for the left indices.
        perm[0] = list(reversed(perm[0]))
    return flatten(perm)


def dims_to_tensor_shape(dims):
    """
    Given the dims of a Qobj instance, returns the shape of the
    corresponding tensor. This helps, for instance, resolve the
    column-stacking convention for superoperators.

    Parameters
    ----------

    dims : list
        Dimensions specification for a Qobj.

    Returns
    -------

    tensor_shape : tuple
        NumPy shape of the corresponding tensor.
    """
    perm = dims_to_tensor_perm(dims)
    dims = flatten(dims)
    return tuple(map(partial(getitem, dims), perm))


def dims_idxs_to_tensor_idxs(dims, indices):
    """
    Given the dims of a Qobj instance, and some indices into
    dims, returns the corresponding tensor indices. This helps
    resolve, for instance, that column-stacking for superoperators,
    oper-ket and oper-bra implies that the input and output tensor
    indices are reversed from their order in dims.

    Parameters
    ----------

    dims : list
        Dimensions specification for a Qobj.

    indices : int, list or tuple
        Indices to convert to tensor indices. Can be specified
        as a single index, or as a collection of indices.
        In the latter case, this can be nested arbitrarily
        deep. For instance, [0, [0, (2, 3)]].

    Returns
    -------

    tens_indices : int, list or tuple
        Container of the same structure as indices containing
        the tensor indices for each element of indices.
    """
    perm = dims_to_tensor_perm(dims)
    return deep_map(partial(getitem, perm), indices)



def _frozen(*args, **kwargs):
    raise RuntimeError("Dimension cannot be modified.")


class MetaSpace(type):
    def __call__(cls, *args, rep='super'):
        if cls is Space and len(args) == 1 and isinstance(args[0], list):
            return cls.from_list(*args, rep=rep)
        elif len(args) == 1 and isinstance(args[0], Space):
            return args[0]

        if len(args) == 0:
            cls = Field
        elif len(args) == 1 and args[0] == 1:
            cls = Field
        elif len(args) == 1 and isinstance(args[0], Dimensions):
            cls = SuperSpace
        elif len(args) > 1:
            if all(isinstance(arg, Field) for arg in args):
                cls = Field
            elif all(isinstance(arg, Space) for arg in args):
                cls = Compound

        args = tuple([tuple(arg) if isinstance(arg, list) else arg
                      for arg in args])

        if cls is Field:
            return cls.field_instance
        if cls is SuperSpace:
            args = *args, rep or 'super'
        if args not in cls._stored_dims:
            instance = cls.__new__(cls)
            instance.__init__(*args)
            cls._stored_dims[args] = instance
        return cls._stored_dims[args]

    def from_list(cls, list_dims, rep='super'):
        if isinstance(list_dims[0], list):
            # Superoperator or tensor of superoperators
            if len(list_dims) % 2 == 0:
                spaces = [
                    Space(Dimensions(
                        Space(list_dims[i+1]),
                        Space(list_dims[i])
                    ), rep=rep)
                    for i in range(0, len(list_dims), 2)
                ]
            elif len(list_dims) == 1:
                spaces = [Space(size) for size in list_dims[0]]
            else:
                raise ValueError(f'Format not understood {list_dims}')
        else:
            spaces = [Space(size) for size in list_dims]
        if len(spaces) == 1:
            return spaces[0]
        elif len(spaces) >= 2:
            return Space(*spaces)
        raise ValueError("Bad list format")


class Space(metaclass=MetaSpace):
    _stored_dims = {}
    def __init__(self, dims):
        if dims <= 0:
            raise ValueError
        # Size of the hilbert space
        self.size = dims
        self.issuper = False
        # Super representation, should be an empty string except for SuperSpace
        self.superrep = ""
        # Does the size and dims match directly: size == prod(dims)
        self._pure_dims = True
        self.__setitem__ = _frozen

    def __repr__(self):
        return f"Space({self.size})"

    def as_list(self):
        return [self.size]

    def __str__(self):
        return str(self.as_list())

    def dims2idx(self, dims):
        return dims

    def idx2dims(self, idx):
        return idx


class Field(Space):
    field_instance = None
    def __init__(self):
        self.size = 1
        self.issuper = False
        self.superrep = ""
        self._pure_dims = True
        self.__setitem__ = _frozen

    def __repr__(self):
        return "Field()"

    def as_list(self):
        return [1]


Field.field_instance = Field.__new__(Field)
Field.field_instance.__init__()


class Compound(Space):
    _stored_dims = {}
    def __init__(self, *spaces):
        self.spaces = []
        for space in spaces:
            if isinstance(space, Compound):
                self.spaces += space.spaces
            else:
                self.spaces += [space]
        self.size = np.prod([space.size for space in self.spaces])
        self.issuper = any(space.issuper for space in self.spaces)
        self._pure_dims = all(space._pure_dims for space in self.spaces)
        superrep = [space.superrep for space in self.spaces]
        if all(superrep[0] == rep for rep in superrep):
            self.superrep = superrep[0]
        else:
            # We could also raise an error
            self.superrep = 'mixed'
        self.__setitem__ = _frozen

    def __repr__(self):
        parts_rep = ", ".join(repr(space) for space in self.spaces)
        return f"Compound({parts_rep})"

    def as_list(self):
        return sum([space.as_list() for space in self.spaces], [])

    def dims2idx(self, dims):
        pos = 0
        step = 1
        for space, dim in zip(self.spaces, dims)[::-1]:
            pos += space.dims2idx(dim) * step
            step *= space.size
        return pos

    def idx2dims(self, idx):
        dims = []
        for space in self.spaces:
            dim, idx = divmod(idx, space.size)
            dims.append(space.idx2dims(dim))
        return dims


class SuperSpace(Space):
    _stored_dims = {}
    def __init__(self, oper, rep='super'):
        self.oper = oper
        self.superrep = rep
        self.size = oper.shape[0] * oper.shape[1]
        self.issuper = True
        self._pure_dims = oper._pure_dims
        self.__setitem__ = _frozen

    def __repr__(self):
        return f"Super({repr(self.oper)}, rep={self.superrep})"

    def as_list(self):
        return self.oper.as_list()

    def dims2idx(self, dims):
        pos = 0
        step = 1
        for space, dim in zip(self.spaces, dims)[::-1]:
            pos += dim * step
            step *= space.size
        return pos

    def idx2dims(self, idx):
        dims = []
        for space in self.spaces:
            dim, idx = divmod(idx, space.size)
            dims.append(dim)
        return dims


class SumSpace(Space):
    """
    Dimensions for piqs's dicke states.
    Example :
        A system of 2 qubit split into a triplet and singulet states:
        SumSpace((3,1))
    """
    _stored_dims = {}
    def __init__(self, structure):
        self.structure = structure
        self.size = sum(structure)
        self.issuper = False
        self.superrep = ""
        self._pure_dims = False
        self.__setitem__ = _frozen

    def __repr__(self):
        return f"SumSpace({self.structure})"

    def as_list(self):
        return list(self.dims)

    def dims2idx(self, dims):
        return dims

    def idx2dims(self, idx):
        return idx


class MetaDims(type):
    def __call__(cls, *args, rep='super'):
        if isinstance(args[0], list):
            args = (
                Space(args[0][1], rep=rep),
                Space(args[0][0], rep=rep)
            )
        elif len(args) == 1 and isinstance(args[0], Dimensions):
            return args[0]
        elif len(args) != 2:
            raise NotImplementedError('No Dual, Ket, Bra...', args)
        elif args[0] == args[1] == Field():
            return Field()

        if args not in cls._stored_dims:
            instance = cls.__new__(cls)
            instance.__init__(*args)
            cls._stored_dims[args] = instance
        return cls._stored_dims[args]


class Dimensions(metaclass=MetaDims):
    _stored_dims = {}
    def __init__(self, from_, to_):
        self.from_ = from_
        self.to_ = to_
        self.shape = to_.size, from_.size
        self.issuper = from_.issuper or to_.issuper
        self._pure_dims = from_._pure_dims and to_._pure_dims
        self.isbra = False
        self.isket = False
        self.isoper = False
        self.isoperbra = False
        self.isoperket = False
        self.issquare = False
        if self.from_ is Field():
            self.type = 'operator-ket' if self.issuper else 'ket'
            self.isket = not self.issuper
            self.isoperket = self.issuper
            self.superrep = self.to_.superrep
        elif self.to_ is Field():
            self.type = 'operator-bra' if self.issuper else 'bra'
            self.isbra = not self.issuper
            self.isoperbra = self.issuper
            self.superrep = self.from_.superrep
        elif self.from_ == self.to_:
            self.type = 'super' if self.issuper else 'oper'
            self.superrep = self.from_.superrep
            self.isoper = not self.issuper
            self.issquare = True
        else:
            self.type = 'super' if self.issuper else 'oper'
            if self.from_.superrep == self.to_.superrep:
                self.superrep = self.from_.superrep
            else:
                self.superrep = 'mixed'
        self.__setitem__ = _frozen


    def __repr__(self):
        return f"Dimensions({repr(self.from_)}, {repr(self.to_)})"

    def __str__(self):
        return str(self.as_list())

    def as_list(self):
        return [self.to_.as_list(), self.from_.as_list()]

    def __getitem__(self, key):
        if key == 0:
            return self.to_
        elif key == 1:
            return self.from_

    def dims2idx(self, dims):
        return self.to_.dims2idx(dims[0]), self.from_.dims2idx(dims[1])

    def idx2dims(self, idx):
        return [self.to_.idx2dims(dims[0]), self.from_.idx2dims(dims[1])]
