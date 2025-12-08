from qutip.core.cy.qobjevo import QobjEvo, _Feedback
from qutip.core.dimensions import Dimensions, Field, SuperSpace
import qutip.core.data as _data
from qutip.core.qobj import Qobj
import numpy as np


class _ExpectFeedback(_Feedback):
    def __init__(self, oper, default=0.):
        self.oper = QobjEvo(oper)
        self.N = oper.shape[1]
        self.N2 = oper.shape[1]**2
        self.default = default

    def check_consistency(self, dims):
        if not (
            self.oper._dims == dims
            or self.oper._dims[1] == dims  # super e_op, oper QobjEvo
            or self.oper._dims == dims[0]  # oper e_op, super QobjEvo
        ):
            raise ValueError(
                f"Dimensions of the expect operator ({self.oper.dims}) "
                f"does not match the operator ({dims})."
            )

    def __call__(self, t, state):
        if state.shape[0] == self.N:
            return self.oper.expect_data(t, state)
        if state.shape[0] == self.N2 and state.shape[1] == 1:
            return self.oper.expect_data(
                t, _data.column_unstack(state, self.N)
            )
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

    def check_consistency(self, dims):
        if not self.open:
            # Ket
            self.dims = Dimensions(Field(), dims[1])
        elif dims.issuper:
            # Density matrix, operator already super
            self.dims = dims[1].oper
        else:
            # operator not super, dm dims match oper
            self.dims = dims
        if self.prop and self.dims[1] == Field():
            self.dims = Dimensions(self.dims[0], self.dims[0])
        elif self.prop:
            self.dims = Dimensions(
                SuperSpace(self.dims), SuperSpace(self.dims)
            )

        if self.default is None:
            pass
        elif self.default._dims == self.dims:
            pass
        elif self.default._dims[0] == self.dims[0]:
            # Catch rectangular state and propagator when the flag is not set.
            self.dims = self.default._dims
        else:
            # The state could not be used for qevo @ default...
            raise TypeError(
                f"The dimensions of the default state ({self.default.dims}) "
                f"does not match the operators ({self.dims})."
            )

    def __call__(self, t, state):
        if state.shape == self.dims.shape:
            out = Qobj(state, dims=self.dims)
        else:
            out = Qobj(
                _data.column_unstack(state, self.dims.shape[0]),
                dims=self.dims
            )
        return out

    def __repr__(self):
        return "QobjFeedback"


class _DataFeedback(_Feedback):
    def __init__(self, default=None, open=True, prop=False):
        self.open = open
        self.default = default
        self.prop = prop

    def check_consistency(self, dims):
        if self.default is None:
            return
        if not (
            dims.shape[1] == self.default.shape[0]
            or (dims.shape[1]**2 == self.default.shape[0] and self.open)
        ):
            raise ValueError(
                f"The shape of the default state {self.default.shape} "
                f"does not match the operators {dims.shape}."
            )
        if self.prop and self.default.shape[0] != self.default.shape[1]:
            raise ValueError(
                f"The default state is expected to be square when computing "
                f"propagators, but is {self.default.shape}."
            )

    def __call__(self, t, state):
        return state

    def __repr__(self):
        return "DataFeedback"


class _CollapseFeedback(_Feedback):
    code = "CollapseFeedback"

    def __init__(self, default=None):
        self.default = default or []

    def check_consistency(self, dims):
        pass

    def __repr__(self):
        return "CollapseFeedback"


def _default_wiener(t):
    return np.zeros(1)


class _WienerFeedback(_Feedback):
    code = "WienerFeedback"

    def __init__(self, default=None):
        self.default = default or _default_wiener

    def check_consistency(self, dims):
        pass

    def __repr__(self):
        return "WienerFeedback"
