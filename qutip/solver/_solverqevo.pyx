#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

from qutip.core.cy.cqobjevo cimport CQobjEvo
from qutip.core.cy.coefficient cimport Coefficient
from qutip.core.data.base cimport idxint
cimport qutip.core.data as _data
from qutip.core.data.reshape cimport column_unstack_dense, column_stack_dense
import qutip.core.data as data
from qutip import Qobj, spre
cimport cython
from qutip.solver._feedback cimport (QobjFeedback, ExpectFeedback,
                                     CollapseFeedback, Feedback)
from qutip.core.data.base import idxint_dtype
from libc.math cimport round
import numpy as np

cdef class SolverQEvo:
    def __init__(self, base, options, dict args, dict feedback):
        self.base_py = base
        self.base = base.compiled_qobjevo
        self.set_feedback(feedback, args, base.cte.issuper,
                          options.ode['feedback_normalize'])
        self.num_calls = 0

    def jac_np_vec(self, t, vec):
        return self.jac_data(t).to_array()

    cpdef _data.Data jac_data(self, double t):
        if self.has_dynamic_args:
            raise NotImplementedError("jacobian not available with feedback")
        self.num_calls += 1
        return self.base.call(t, data=True)

    def mul_np_double_vec(self, t, vec):
        vec_cplx = vec.view(complex)
        return self.mul_np_vec(t, vec_cplx).view(np.float64)

    def mul_np_vec(self, t, vec):
        cdef int i, row, col
        cdef _data.Dense state = _data.dense.fast_from_numpy(vec)
        column_unstack_dense(state, self.base.shape[1], inplace=True)
        cdef _data.Dense out = _data.dense.zeros(state.shape[0],
                                state.shape[1], state.fortran)
        out = self.mul_dense(t, state, out)
        column_stack_dense(out, inplace=True)
        return out.as_ndarray().ravel()

    cpdef _data.Data mul_data(self, double t, _data.Data vec,
                              _data.Data out=None):
        if (
            type(vec) is _data.Dense and
            (out is None or type(out) is _data.Dense)
        ):
            return self.mul_dense(t, vec, out)
        if self.has_dynamic_args:
            self.apply_feedback(t, vec)
        self.num_calls += 1
        if out is None:
            return self.base.matmul(t, vec)
        else:
            return data.add(out, self.base.matmul(t, vec))

    cpdef _data.Dense mul_dense(self, double t, _data.Dense vec,
                                _data.Dense out=None):
        if self.has_dynamic_args:
            self.apply_feedback(t, vec)
        self.num_calls += 1
        return self.base.matmul_dense(t, vec, out)

    def set_feedback(self, dict feedback, dict args, bint issuper, bint norm):
        # Move elsewhere and op should be a dimensions object when available
        self.args = args
        self.feedback = []
        for key, val in feedback.items():
            if val in [Qobj, "Qobj", "qobj", "state"]:
                self.feedback.append(QobjFeedback(key, args[key],
                                                           norm))
            elif isinstance(val, Qobj):
                self.feedback.append(ExpectFeedback(key, val,
                                                             issuper, norm))
            elif val in ["collapse"]:
                self.feedback.append(
                    CollapseFeedback(key)
                )
            else:
                raise ValueError("unknown feedback type")
        self.has_dynamic_args = bool(self.feedback)

    def update_feedback(self, what, data):
        for dargs in self.feedback:
            dargs.update(what, data)

    cpdef void apply_feedback(self, double t, _data.Data matrix) except *:
        cdef Feedback feedback
        for dyn_args in self.feedback:
            feedback = <Feedback> dyn_args
            val = feedback._call(t, matrix)
            self.args[feedback.key] = val
        self.base_py.arguments(self.args)

    cpdef void arguments(self, dict args):
        self.args = args
        self.base_py.arguments(args)

    @property
    def stats(self):
        return {"num_calls": self.num_calls}
