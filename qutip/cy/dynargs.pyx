




# ToDo for v5:
# Use data layer

cdef class DynArgs:
    """
    Class for arguments that change depending on the state.
    The main call is:
    DynArgs(state) => key, val
    """
    def __init__(self):
        pass

    cdef object _call(selt, complex[:] state) except *:
        # Entry point for Array
        return None

    def __call__(self, state):
        # Entry point for Qobj
        return self._call(state.full().ravel("F"))


cdef class StateArray(DynArgs):
    def __init__(self, key):
        self.key = key

    cdef object _call(selt, complex[:] state) except *:
        return self.key, np.array(state)


cdef class StateMat(DynArgs):
    def __init__(self, key, mat):
        self.key = key
        self.shape0 = mat.shape[0]
        self.shape1 = mat.shape[1]

    cdef object _call(selt, complex[:] state) except *:
        return self.key, np.array(state).reshape(self.shape0, self.shape1)


cdef class StateQobj(DynArgs):
    def __init__(self, key, qobj):
        self.shape0 = qobj.shape[0]
        self.shape1 = qobj.shape[1]
        self.dims = qobj.dims
        self.key = key

    cdef object _call(selt, complex[:] state) except *:
        arr = np.array(state).reshape(self.shape0, self.shape1)
        return self.key, Qobj(arr, dims=self.dims)


cdef class StateExpect(DynArgs):
    def __init__(self, key, qoe):
        cdef CQobjEvo self.expect_op = qoe.compiled_qobjevo
        self.key = key

    cdef object _call(selt, complex[:] state) except *:
        cdef complex out
        cdef int nn = state.shape[0] * state.shape[1]
        if self.expect_op.shape1 != nn:
            out = self.expect_op._overlapse(t, &state[0])
        elif self.expect_op.super:
            out = self.expect_op._expect_super(t, &state[0])
        else:
            out = self.expect_op._expect(t, &state[0])
        return self.key, out
