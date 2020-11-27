#cython: language_level=3
from qutip import Qobj, spre
from qutip.core import data
from qutip.core.data.mul import mul_inplace, mul
from qutip.core cimport data as _data

cdef class Feedback:
    def __init__(self, key, state):
        self.key = key

    cdef object _call(self, double t, _data.Data state):
        return 0


cdef class QobjFeedback(Feedback):
    def __init__(self, key, state, norm):
        self.norm = norm
        self.key = key
        self.dims = state.dims
        self.type = state.type
        self.superrep = state.superrep
        self.isherm = state.isherm
        self.isunitary = state.isunitary
        self.shape = state.shape
        self.issuper = not state.isket

    cdef object _call(self, double t, _data.Data state):
        cdef _data.Data matrix = data.reshape(state,
                                              self.shape[0], self.shape[1])
        if self.norm:
            if self.issuper:
                mul_inplace(matrix, 1 / data.norm.trace(matrix))
            else:
                mul_inplace(matrix, 1 / data.norm.l2(matrix))
        return Qobj(arg=matrix, dims=self.dims,
                    type=self.type, copy=False,
                    superrep=self.superrep, isherm=self.isherm,
                    isunitary=self.isunitary)


cdef class ExpectFeedback(Feedback):
    def __init__(self, key, op, issuper, norm):
        self.key = key
        self.norm = norm
        if issuper:
            self.op = spre(op).data
        else:
            self.op = op.data
        self.issuper = issuper

    cdef object _call(self, double t, _data.Data state):
        if self.norm:
            if self.issuper:
                state = mul(state, 1 / data.norm.trace(state))
            else:
                state = mul(state, 1 / data.norm.l2(state))
        if self.issuper:
            return data.expect_super(self.op, state)
        return data.expect(self.op, state)


cdef class CollapseFeedback(Feedback):
    def __init__(self, key):
        self.key = key
        self.collapse = []

    cdef object _call(self, double t, _data.Data state):
        return self.collapse

    def set_collapse(self, collapse):
        self.collapse = collapse
