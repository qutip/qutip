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
        return self.jac_data(t).as_array()

    cdef _data.Data jac_data(self, double t):
        if self.has_dynamic_args:
            raise NotImplementedError("jacobian not available with feedback")
        self.num_calls += 1
        return self.base.call(t, data=True)

    cpdef list get_coeff(self, double t, vec=None):
        cdef Feedback feedback
        cdef _data.Dense state
        if self.has_dynamic_args and vec is not None:
            state = _data.dense.fast_from_numpy(vec)
            for dyn_args in self.dynamic_arguments:
                feedback = <Feedback> dyn_args
                val = feedback._call(t, state)
                self.args[feedback.key] = val
            for i in range(self.base.n_ops):
                (<Coefficient> self.base.coeff[i]).arguments(self.args)
        out = []
        for i in range(self.base.n_ops):
            out.append((<Coefficient> self.base.coeff[i])(t))
        return out

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

    cdef _data.Data mul_data(self, double t, _data.Data vec):
        if self.has_dynamic_args:
            self.apply_feedback(t, vec)
        self.num_calls += 1
        return self.base.matmul(t, vec)

    cdef _data.Dense mul_dense(self, double t, _data.Dense vec, _data.Dense out):
        if self.has_dynamic_args:
            self.apply_feedback(t, vec)
        self.num_calls += 1
        return self.base.matmul_dense(t, vec, out)

    def set_feedback(self, dict feedback, dict args, bint issuper, bint norm):
        # Move elsewhere and op should be a dimensions object when available
        self.args = args
        self.dynamic_arguments = []
        for key, val in feedback.items():
            if val in [Qobj, "Qobj", "qobj", "state"]:
                self.dynamic_arguments.append(QobjFeedback(key, args[key],
                                                           norm))
            elif isinstance(val, Qobj):
                self.dynamic_arguments.append(ExpectFeedback(key, val,
                                                             issuper, norm))
            elif val in ["collapse"]:
                self.dynamic_arguments.append(
                    CollapseFeedback(key)
                )
            else:
                raise ValueError("unknown feedback type")
        self.has_dynamic_args = bool(self.dynamic_arguments)

    def update_feedback(self, collapse):
        for dargs in self.dynamic_arguments:
            if isinstance(dargs, CollapseFeedback):
                dargs.set_collapse(collapse)

    cpdef void apply_feedback(self, double t, _data.Data matrix) except *:
        cdef Feedback feedback
        for dyn_args in self.dynamic_arguments:
            feedback = <Feedback> dyn_args
            val = feedback._call(t, matrix)
            self.args[feedback.key] = val
        for i in range(self.base.n_ops):
            (<Coefficient> self.base.coeff[i]).arguments(self.args)

    cpdef void arguments(self, dict args):
        self.base_py.arguments(args)

    @property
    def stats(self):
        return {"num_calls": self.num_calls}
