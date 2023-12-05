from qutip.core.data.constant import zeros
from qutip.core.cy.qobjevo import QobjEvo, _Feedback
from qutip.core.dimensions import Dimensions
import numpy as np

def _expected_feedback_dims(dims, dm, prop):
    """
    Return the state expected dimension from the operator dimension

    dims : dims of the operator
    dm : if the state is a dm or ket
    prop : if the evolution evolve state or propagator.
    """
    if not self.open:
        # Ket
        dims = Dimensions(dims[1], Field())
    elif dims.issuper:
        # Density matrix, operator already super
        dims = dims[1].oper
    else:
        # operator not super, dm dims match oper
        dims = dims
    if not prop:
        return dims
    elif dims.isket:
        return Dimensions(dims[0], dims[0])
    else:
        return Dimensions(dims, dims)


class _ExpectFeedback(_Feedback):
    def __init__(self, oper, default=None):
        self.oper = QobjEvo(oper)
        self.N = oper.shape[1]
        self.N2 = oper.shape[1]**2
        self.default = default

    def prepare(self, dims):
        if not (
            self.oper._dims == dims
            or self.oper._dims[1] == dims
        ):
            raise ValueError(
                f"Dimensions of the expect operator ({self.oper.dims}) "
                f"does not match the operator ({dims})."
            )
        return self.default or 0.

    def __call__(self, t, state):
        if state.shape[0] == self.N:
            return self.oper.expect_data(t, state)
        if state.shape[0] == self.N2 and state.shape[1] == 1:
            return self.oper.expect_data(t, _data.column_unstack(state, self.N))
        raise ValueError(
            f"Shape of the expect operator ({self.oper.shape}) "
            f"does not match the state ({state.shape})."
        )

    def __repr__(self):
        return "ExpectFeedback"


class _QobjFeedback(_Feedback):
    def __init__(self, default=None, prop=False, open=True):
        self.open = open
        self.prop = prop
        self.default = default

    def prepare(self, dims):
        self.dims = _expected_feedback_dims(dims, self.open, self.prop)

        if self.default is not None and self.default.dims == self.dims:
            return self.default
        elif self.default is not None and self.default.dims[0] == self.dims[0]:
            # Catch rectangular state and propagator when prop is not set.
            self.dims = self.default.dims
            return self.default

        return qzero(dims)

    def __call__(self, t, state):
        if state.shape == self.dims.shape:
            out = Qobj(state, dims=self.dims)
        else:
            out = Qobj(reshape_dense(state, *self.dims.shape), dims=self.dims)

        return out

    def __repr__(self):
        return "QobjFeedback"


class _DataFeedback(_Feedback):
    def __init__(self, default=None, open=True, prop=False):
        self.open = open
        self.default = default
        self.prop = prop

    def prepare(self, dims):
        if self.default is not None:
            return self.default
        dims = _expected_feedback_dims(dims, self.open, self.prop)
        if not self.prop:
            return zeros[self.dtype](np.prod(dims.shape), 1)
        return zeros["csr"](*dims.shape)

    def __call__(self, t, state):
        return state

    def __repr__(self):
        return "DataFeedback"


class _CollapseFeedback(_Feedback):
    code = "CollapseFeedback"

    def __init__(self, default=None):
        self.default = default

    def prepare(self, dims):
        return self.default or []

    def __repr__(self):
        return "CollapseFeedback"


def _default_weiner(t):
    return np.zeros(1)

class _WeinerFeedback(_Feedback):
    code = "WeinerFeedback"

    def __init__(self, default=None):
        self.default = default

    def prepare(self, dims):
        return self.default or _default_weiner

    def __repr__(self):
        return "WeinerFeedback"
