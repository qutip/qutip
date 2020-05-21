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
"""Time-dependent Quantum Object (Qobj) class.
"""
__all__ = ['QobjEvo']

from qutip.qobj import Qobj
import qutip.settings as qset
from qutip.interpolate import Cubic_Spline
from scipy.interpolate import CubicSpline, interp1d
from functools import partial, wraps
from types import FunctionType, BuiltinFunctionType
import numpy as np
from numbers import Number
from qutip.qobjevo_codegen import (_compile_str_single, _compiled_coeffs,
                                   _compiled_coeffs_python)
from qutip.cy.spmatfuncs import (cy_expect_rho_vec, cy_expect_psi,
                                 spmv, cy_spmm_tr)
import atexit
import pickle
import sys
import scipy
import os
from re import sub

safePickle = [False]
if sys.platform == 'win32':
    safePickle[0] = True


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# object for each time dependent element of the QobjEvo
# qobj : the Qobj of element ([*Qobj*, f])
# get_coeff : a callable that take (t, args) and return the coeff at that t
# coeff : The coeff as a string, array or function as provided by the user.
# type : flag for the type of coeff
"""
class EvoElement:
    def __init__(self, qobj, get_coeff, coeff, type):
        self.qobj = qobj
        self.get_coeff = get_coeff
        self.coeff = coeff
        self.type = type

    @classmethod
    def make(cls, list_):
        return cls(*list_)

    def __getitem__(self, i):
        if i == 0:
            return self.qobj
        if i == 1:
            return self.get_coeff
        if i == 2:
            return self.coeff
        if i == 3:
            return self.type
"""

EvoElement = namedtuple("EvoElement", "qobj coeff")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class StateArgs:
    """Object to indicate to use the state in args outside solver.
    args[key] = StateArgs(type, op)
    """
    def __init__(self, type="Qobj", op=None):
        self.dyn_args = (type, op)

    def __call__(self):
        return self.dyn_args
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class QobjEvo:
    """A class for representing time-dependent quantum objects,
    such as quantum operators and states.

    The QobjEvo class is a representation of time-dependent Qutip quantum
    objects (Qobj). This class implements math operations :
        +,- : QobjEvo, Qobj
        * : Qobj, C-number
        / : C-number
    and some common linear operator/state operations. The QobjEvo
    are constructed from a nested list of Qobj with their time-dependent
    coefficients. The time-dependent coefficients are either a funciton, a
    string or a numpy array.

    For function format, the function signature must be f(t, args).
    *Examples*
        def f1_t(t, args):
            return np.exp(-1j * t * args["w1"])

        def f2_t(t, args):
            return np.cos(t * args["w2"])

        H = QobjEvo([H0, [H1, f1_t], [H2, f2_t]], args={"w1":1., "w2":2.})

    For string based coeffients, the string must be a compilable python code
    resulting in a complex. The following symbols are defined:
        sin cos tan asin acos atan pi
        sinh cosh tanh asinh acosh atanh
        exp log log10 erf zerf sqrt
        real imag conj abs norm arg proj
        numpy as np, and scipy.special as spe.
    *Examples*
        H = QobjEvo([H0, [H1, 'exp(-1j*w1*t)'], [H2, 'cos(w2*t)']],
                    args={"w1":1.,"w2":2.})

    For numpy array format, the array must be an 1d of dtype float or complex.
    A list of times (float64) at which the coeffients must be given (tlist).
    The coeffients array must have the same len as the tlist.
    The time of the tlist do not need to be equidistant, but must be sorted.
    By default, a cubic spline interpolation will be used for the coefficient
    at time t.
    If the coefficients are to be treated as step function, use the arguments
    args = {"_step_func_coeff": True}
    *Examples*
        tlist = np.logspace(-5,0,100)
        H = QobjEvo([H0, [H1, np.exp(-1j*tlist)], [H2, np.cos(2.*tlist)]],
                    tlist=tlist)

    args is a dict of (name:object). The name must be a valid variables string.
    Some solvers support arguments that update at each call:
    sesolve, mesolve, mcsolve:
        state can be obtained with:
            "state_vec":psi0, args["state_vec"] = state as 1D np.ndarray
            "state_mat":psi0, args["state_mat"] = state as 2D np.ndarray
            "state":psi0, args["state"] = state as Qobj

            This Qobj is the initial value.

        expectation values:
            "expect_op_n":0, args["expect_op_n"] = expect(e_ops[int(n)], state)
            expect is <phi|O|psi> or tr(state * O) depending on state dimensions

    mcsolve:
        collapse can be obtained with:
            "collapse":list => args[name] == list of collapse
            each collapse will be appended to the list as (time, which c_ops)

    Mixing the formats is possible, but not recommended.
    Mixing tlist will cause problem.

    Parameters
    ----------
    QobjEvo(Q_object=[], args={}, tlist=None)

    Q_object : array_like
        Data for vector/matrix representation of the quantum object.

    args : dictionary that contain the arguments for

    tlist : array_like
        List of times at which the numpy-array coefficients are applied. Times
        must be equidistant and start from 0.

    Attributes
    ----------
    cte : Qobj
        Constant part of the QobjEvo

    ops : list of EvoElement
        List of Qobj and the coefficients.
        [(Qobj, coefficient as a function, original coefficient,
            type, local arguments ), ... ]
        type :
            1: function
            2: string
            3: np.array
            4: Cubic_Spline

    args : map
        arguments of the coefficients

    dynamics_args : list
        arguments that change during evolution

    tlist : array_like
        List of times at which the numpy-array coefficients are applied.

    compiled : string
        Has the cython version of the QobjEvo been created

    compiled_qobjevo : cy_qobj (CQobjCte or CQobjEvoTd)
        Cython version of the QobjEvo

    coeff_get : callable object
        object called to obtain a list of coefficient at t

    coeff_files : list
        runtime created files to delete with the instance

    dummy_cte : bool
        is self.cte a empty Qobj

    const : bool
        Indicates if quantum object is Constant

    type : string
        information about the type of coefficient
            "string", "func", "array",
            "spline", "mixed_callable", "mixed_compilable"

    num_obj : int
        number of Qobj in the QobjEvo : len(ops) + (1 if not dummy_cte)

    use_cython : bool
        flag to compile string to cython or python

    safePickle : bool
        flag to not share pointers between thread


    Methods
    -------
    copy() :
        Create copy of Qobj

    arguments(new_args):
        Update the args of the object

    Math:
        +/- QobjEvo, Qobj, scalar:
            Addition is possible between QobjEvo and with Qobj or scalar
        -:
            Negation operator
        * Qobj, scalar:
            Product is possible with Qobj or scalar
        / scalar:
            It is possible to divide by scalar only
    conj()
        Return the conjugate of quantum object.

    dag()
        Return the adjoint (dagger) of quantum object.

    trans()
        Return the transpose of quantum object.

    _cdc()
        Return self.dag() * self.

    permute(order)
        Returns composite qobj with indices reordered.

    apply(f, *args, **kw_args)
        Apply the function f to every Qobj. f(Qobj) -> Qobj
        Return a modified QobjEvo and let the original one untouched

    apply_decorator(decorator, *args, str_mod=None,
                    inplace_np=False, **kw_args):
        Apply the decorator to each function of the ops.
        The *args and **kw_args are passed to the decorator.
        new_coeff_function = decorator(coeff_function, *args, **kw_args)
        str_mod : list of 2 elements
            replace the string : str_mod[0] + original_string + str_mod[1]
            *exemple: str_mod = ["exp(",")"]
        inplace_np:
            Change the numpy array instead of applying the decorator to the
            function reading the array. Some decorators create incorrect array.
            Transformations f'(t) = f(g(t)) create a missmatch between the
            array and the associated time list.

    tidyup(atol=1e-12)
        Removes small elements from quantum object.

    compress():
        Merge ops which are based on the same quantum object and coeff type.

    compile(code=False, matched=False, dense=False, omp=0):
        Create the associated cython object for faster usage.
        code: return the code generated for compilation of the strings.
        matched: the compiled object use sparse matrix with matching indices.
                    (experimental, no real advantage)
        dense: the compiled object use dense matrix.
        omp: (int) number of thread: the compiled object use spmvpy_openmp.

    __call__(t, data=False, state=None, args={}):
        Return the Qobj at time t.
        *Faster after compilation

    mul_mat(t, mat):
        Product of this at t time with the dense matrix mat.
        *Faster after compilation

    mul_vec(t, psi):
        Apply the quantum object (if operator, no check) to psi.
        More generaly, return the product of the object at t with psi.
        *Faster after compilation

    expect(t, psi, herm=False):
        Calculates the expectation value for the quantum object (if operator,
            no check) and state psi.
        Return only the real part if herm.
        *Faster after compilation

    to_list():
        Return the time-dependent quantum object as a list
    """

    def __init__(self, Q_object=[], args={}, copy=True,
                 tlist=None, state0=None, e_ops=[]):
        if isinstance(Q_object, QobjEvo):
            if copy:
                self._inplace_copy(Q_object)
            else:
                self.__dict__ = Q_object.__dict__
            if args:
                self.arguments(args)
                for i, dargs in enumerate(self.dynamics_args):
                    e_int = dargs[1] == "expect" and isinstance(dargs[2], int)
                    if e_ops and e_int:
                        self.dynamics_args[i] = (dargs[0], "expect",
                                                 e_ops[dargs[2]])
                if state0 is not None:
                    self._dynamics_args_update(0., state0)
            return

        self.const = False
        self.dummy_cte = False
        self.args = args.copy()
        self.dynamics_args = []
        self.cte = None
        self.tlist = tlist
        self.compiled = ""
        self.compiled_qobjevo = None
        self.coeff_get = None
        self.type = "none"
        self.omp = 0
        self.use_cython = use_cython[0]
        self.safePickle = safePickle[0]

        if isinstance(Q_object, list) and len(Q_object) == 2:
            if isinstance(Q_object[0], Qobj) and not isinstance(Q_object[1],
                                                                (Qobj, list)):
                # The format is [Qobj, coeff]
                Q_object = [Q_object]

        op_type = self._td_format_check_single(Q_object, tlist)
        self.ops = []

        if isinstance(Q_object, Qobj):
            self.cte = Q_object
            self.const = True
            self.type = "cte"
        else:
            for op in Q_object:
                if isinstance(op, Qobj):
                    if self.cte is None:
                        self.cte = op
                    else:
                        self.cte += op
                else:
                    self.ops.append(EvoElement(op[0], Coeff(op[1])))

            if self.cte is None:
                self.cte = self.ops[0].qobj * 0
                self.dummy_cte = True

            try:
                cte_copy = self.cte.copy()
                # test is all qobj are compatible (shape, dims)
                for op in self.ops:
                    cte_copy += op.qobj
            except Exception as e:
                raise TypeError("Qobj not compatible.") from e

            if not self.ops:
                self.const = True
        self._args_checks()
        if e_ops:
            for i, dargs in enumerate(self.dynamics_args):
                if dargs[1] == "expect" and isinstance(dargs[2], int):
                    self.dynamics_args[i] = (dargs[0], "expect",
                                             QobjEvo(e_ops[dargs[2]]))
        if state0 is not None:
            self._dynamics_args_update(0., state0)

    def _args_checks(self):
        statedims = [self.cte.dims[1],[1]]
        for key in self.args:
            if key == "state" or key == "state_qobj":
                self.dynamics_args += [(key, "Qobj", None)]
                if self.args[key] is None:
                    self.args[key] = Qobj(dims=statedims)

            if key == "state_mat":
                self.dynamics_args += [("state_mat", "mat", None)]
                if isinstance(self.args[key], Qobj):
                    self.args[key] = self.args[key].full()
                if self.args[key] is None:
                    self.args[key] = Qobj(dims=statedims).full()

            if key == "state_vec":
                self.dynamics_args += [("state_vec", "vec", None)]
                if isinstance(self.args[key], Qobj):
                    self.args[key] = self.args[key].full().ravel("F")
                if self.args[key] is None:
                    self.args[key] = Qobj(dims=statedims).full().ravel("F")

            if key.startswith("expect_op_"):
                e_op_num = int(key[10:])
                self.dynamics_args += [(key, "expect", e_op_num)]

            if isinstance(self.args[key], StateArgs):
                self.dynamics_args += [(key, *self.args[key]())]
                self.args[key] = 0.

    def _check_old_with_state(self):
        add_vec = False
        for op in self.ops:
            add_vec += op.coeff._check_old_with_state()
        if add_vec:
            self.dynamics_args += [("_state_vec", "vec", None)]

    def __call__(self, t, data=False, state=None, args={}):
        try:
            t = float(t)
        except Exception as e:
            raise TypeError("t should be a real scalar.") from e

        if state is not None:
            self._dynamics_args_update(t, state)

        if args:
            if not isinstance(args, dict):
                raise TypeError("The new args must be in a dict")
            old_args = self.args.copy()
            self.arguments(args)
            op_t = self.__call__(t, data=data)
            self.arguments(old_args)
        elif self.const:
            if data:
                op_t = self.cte.data.copy()
            else:
                op_t = self.cte.copy()
        elif self.compiled and self.compiled != "dense":
            op_t = self.compiled_qobjevo.call(t, data)
        elif data:
            op_t = self.cte.data.copy()
            for part in self.ops:
                op_t += part.qobj.data * part.coeff(t, self.args)
        else:
            op_t = self.cte.copy()
            for part in self.ops:
                op_t += part.qobj * part.coeff(t, self.args)

        return op_t

    def _dynamics_args_update(self, t, state):
        # TO-DO dyn_args class
        if isinstance(state, Qobj):
            for name, what, op in self.dynamics_args:
                if what == "vec":
                    self.args[name] = state.full().ravel("F")
                elif what == "mat":
                    self.args[name] = state.full()
                elif what == "Qobj":
                    self.args[name] = state
                elif what == "expect":
                    self.args[name] = op.expect(t, state)

        elif isinstance(state, np.ndarray) and state.ndim == 1:
            s1 = self.cte.shape[1]
            for name, what, op in self.dynamics_args:
                if what == "vec":
                    self.args[name] = state
                elif what == "expect":
                    self.args[name] = op.expect(t, state)
                elif state.shape[0] == s1 and self.cte.issuper:
                    new_l = int(np.sqrt(s1))
                    mat = state.reshape((new_l, new_l), order="F")
                    if what == "mat":
                        self.args[name] = mat
                    elif what == "Qobj":
                        self.args[name] = Qobj(mat, dims=self.cte.dims[1])
                elif state.shape[0] == s1:
                    mat = state.reshape((-1,1))
                    if what == "mat":
                        self.args[name] = mat
                    elif what == "Qobj":
                        self.args[name] = Qobj(mat, dims=[self.cte.dims[1],[1]])
                elif state.shape[0] == s1*s1:
                    new_l = int(np.sqrt(s1))
                    mat = state.reshape((new_l, new_l), order="F")
                    if what == "mat":
                        self.args[name] = mat
                    elif what == "Qobj":
                        self.args[name] = Qobj(mat, dims=[self.cte.dims[1],
                                                          self.cte.dims[1]])

        elif isinstance(state, np.ndarray) and state.ndim == 2:
            s1 = self.cte.shape[1]
            new_l = int(np.sqrt(s1))
            for name, what, op in self.dynamics_args:
                if what == "vec":
                    self.args[name] = state.ravel("F")
                elif what == "mat":
                    self.args[name] = state
                elif what == "expect":
                    self.args[name] = op.expect(t, state)
                elif state.shape[1] == 1:
                    self.args[name] = Qobj(state, dims=[self.cte.dims[1],[1]])
                elif state.shape[1] == s1:
                    self.args[name] = Qobj(state, dims=self.cte.dims)
                else:
                    self.args[name] = Qobj(state)

        else:
            raise TypeError("state must be a Qobj or np.ndarray")

    def copy(self):
        new = QobjEvo(self.cte.copy())
        new.const = self.const
        new.args = self.args.copy()
        new.dynamics_args = self.dynamics_args.copy()
        new.tlist = self.tlist
        new.dummy_cte = self.dummy_cte
        new.num_obj = self.num_obj
        new.type = self.type
        new.compiled = False
        new.compiled_qobjevo = None
        if self.coeff_get is not None:
            new.coeff_get = self.coeff_get.copy()
        else:
            new.coeff_get = None
        new.coeff_files = []
        new.use_cython = self.use_cython
        new.safePickle = self.safePickle

        for op in self.ops:
            new.ops.append( EvoElement(op.qobj.copy(), op.coeff.copy()) )

        return new

    def _inplace_copy(self, other):
        self.cte = other.cte
        self.const = other.const
        self.args = other.args.copy()
        self.dynamics_args = other.dynamics_args
        self.tlist = other.tlist
        self.dummy_cte = other.dummy_cte
        self.num_obj = other.num_obj
        self.type = other.type
        self.compiled = ""
        self.compiled_qobjevo = None
        if other.coeff_get is not None:
            self.coeff_get = other.coeff_get.copy()
        else:
            self.coeff_get = None
        self.ops = []
        self.use_cython = other.use_cython
        self.safePickle = other.safePickle

        for op in other.ops:
            self.ops.append( EvoElement(op.qobj.copy(), op.coeff.copy()) )

    def arguments(self, new_args):
        if not isinstance(new_args, dict):
            raise TypeError("The new args must be in a dict")
        # remove dynamics_args that are to be refreshed
        self.dynamics_args = [dargs for dargs in self.dynamics_args
                              if dargs[0] not in new_args]
        self.args.update(new_args)
        self._args_checks()
        if self.coeff_get is not None:
            self.coeff_get.set_args(self.args, self.dynamics_args)

    def solver_set_args(self, new_args, state, e_ops):
        self.dynamics_args = []
        self.args.update(new_args)
        self._args_checks()
        for i, dargs in enumerate(self.dynamics_args):
            if dargs[1] == "expect" and isinstance(dargs[2], int):
                self.dynamics_args[i] = (dargs[0], "expect",
                                         QobjEvo(e_ops[dargs[2]]))
                if self.compiled:
                    self.dynamics_args[i][2].compile()
        self._dynamics_args_update(0., state)
        if self.coeff_get is not None:
            self.coeff_get.set_args(self.args, self.dynamics_args)

    def to_list(self):
        list_qobj = []
        if not self.dummy_cte:
            list_qobj.append(self.cte)
        for op in self.ops:
            list_qobj.append([op.qobj, op.coeff])
        return list_qobj

    @property
    def num_ops(self):
        return len(self.ops) + (0 if self.dummy_cte else 1)

    # Math function
    def __add__(self, other):
        res = self.copy()
        res += other
        return res

    def __radd__(self, other):
        res = self.copy()
        res += other
        return res

    def __iadd__(self, other):
        if isinstance(other, QobjEvo):
            self.cte += other.cte
            for op in other.ops:
                self.ops.append(EvoElement(op.qobj.copy(),
                                           op.get_coeff.copy() ))
            self.args.update(**other.args)
            self.dynamics_args += other.dynamics_args
            self.const = self.const and other.const
            self.dummy_cte = self.dummy_cte and other.dummy_cte
            self.compiled = ""
            self.compiled_qobjevo = None
            self.coeff_get = None
        else:
            self.cte += other
            self.dummy_cte = False
            self._recompile()

        return self

    def __sub__(self, other):
        res = self.copy()
        res -= other
        return res

    def __rsub__(self, other):
        res = -self.copy()
        res += other
        return res

    def __isub__(self, other):
        self += (-other)
        return self

    def __mul__(self, other):
        res = self.copy()
        res *= other
        return res

    def __rmul__(self, other):
        res = self.copy()
        if isinstance(other, Qobj):
            res.cte = other * res.cte
            for op in res.ops:
                op.qobj = other * op.qobj
            self._recompile()
            return res
        else:
            res *= other
            return res

    def __imul__(self, other):
        if isinstance(other, Qobj) or isinstance(other, Number):
            self.cte *= other
            for op in self.ops:
                op.qobj *= other
        elif isinstance(other, QobjEvo):
            if other.const:
                self.cte *= other.cte
                for op in self.ops:
                    op.qobj *= other.cte
            elif self.const:
                cte = self.cte.copy()
                self = other.copy()
                self.cte = cte * self.cte
                for op in self.ops:
                    op.qobj = cte * op.qobj
            else:
                cte = self.cte.copy()
                self.cte *= other.cte
                new_terms = []
                old_ops = self.ops
                if not other.dummy_cte:
                    for op in old_ops:
                        new_terms.append(EvoElement(op.qobj * other.cte,
                                                    op.coeff))
                if not self.dummy_cte:
                    for op in other.ops:
                        new_terms.append(EvoElement(cte * op.qobj,
                                                    op.coeff))
                for op_L in old_ops:
                    for op_R in other.ops:
                        new_terms.append(EvoElement(op_L.qobj * op_R.qobj,
                                                    op_L.coeff * op_R.coeff))
                self.ops = new_terms
                self.args.update(other.args)
                self.dynamics_args += other.dynamics_args
                self.dummy_cte = self.dummy_cte and other.dummy_cte

        else:
            raise TypeError("QobjEvo can only be multiplied"
                            " with QobjEvo, Qobj or numbers")
        self._recompile()
        return self

    def __div__(self, other):
        if isinstance(other, (int, float, complex,
                              np.integer, np.floating, np.complexfloating)):
            res = self.copy()
            res *= other**(-1)
            return res
        else:
            raise TypeError('Incompatible object for division')

    def __idiv__(self, other):
        if isinstance(other, (int, float, complex,
                              np.integer, np.floating, np.complexfloating)):
            self *= other**(-1)
        else:
            raise TypeError('Incompatible object for division')
        return self

    def __truediv__(self, other):
        return self.__div__(other)

    def __neg__(self):
        res = self.copy()
        res.cte = -res.cte
        res.ops = [EvoElement(-op.qobj, op.coeff) for op in res.ops]
        res._recompile()
        return res

    # Transformations
    def trans(self):
        res = self.copy()
        res.cte = res.cte.trans()
        res.ops = [EvoElement(op.qobj.trans(), op.coeff)
                   for op in self.ops)]
        res._recompile()
        return res

    def conj(self):
        res = self.copy()
        res.cte = res.cte.conj()
        res.ops = [EvoElement(op.qobj.conj(), op.coeff.conj())
                   for op in self.ops)]
        return res

    def dag(self):
        res = self.copy()
        res.cte = res.cte.dag()
        res.ops = [EvoElement(op.qobj.dag(), op.coeff.conj())
                   for op in self.ops)]
        return res

    def _cdc(self):
        """return a.dag * a """
        if not self.num_obj == 1:
            res = self.dag()
            res *= self
        else:
            res = self.copy()
            res.cte = res.cte.dag() * res.cte
            res.ops = [EvoElement(op.qobj.dag() * op.qobj, op.coeff._cdc())
                       for op in self.ops)]
        return res

    def _shift(self):
        self.compiled = ""
        self.coeff_get = None
        self.args.update({"_t0": 0})
        self.ops = [EvoElement(op.qobj, op.coeff._shift()) for op in self.ops)]
        return self

    # Unitary function of Qobj
    def tidyup(self, atol=1e-12):
        self.cte = self.cte.tidyup(atol)
        for op in self.ops:
            op.qobj = op.qobj.tidyup(atol)
        return self

    def _compress_make_set(self):
        sets = []
        callable_flags = ["func", "spline"]
        for i, op1 in enumerate(self.ops):
            already_matched = any(i in _set for _set in sets)
            if not already_matched:
                this_set = [i]
                for j, op2 in enumerate(self.ops[i+1:]):
                    if op1.qobj == op2.qobj:
                        this_set.append(j+i+1)
                sets.append(this_set)

        fsets = []
        for i, op1 in enumerate(self.ops):
            already_matched = any(i in _set for _set in fsets)
            if not already_matched:
                this_set = [i]
                for j, op2 in enumerate(self.ops[i+1:]):
                    if op1.coeff is op2.coeff:
                        this_set.append(j+i+1)
                fsets.append(this_set)
        return sets, fsets

    def _compress_merge_qobj(self, sets):
        callable_flags = ["func", "spline"]
        new_ops = []
        for _set in sets:
            if len(_set) == 1:
                new_ops.append(self.ops[_set[0]])

            else:
                # Would be nice but need to defined addition with 0 (int)
                # new_coeff = sum(self.ops[i].coeff for i in _set)
                new_coeff = self.ops[_set[0]].coeff
                for i in _set:
                    new_coeff += self.ops[i].coeff
                new_ops.append(EvoElement(self.ops[_set[0]].qobj, new_coeff))

        self.ops = new_ops

    def _compress_merge_func(self, fsets):
        new_ops = []
        for _set in fsets:
            base = self.ops[_set[0]]
            new_op = [None, base.coeff]
            new_op[0] += sum(self.ops[i].qobj for

            if len(_set) == 1:
                new_op[0] = base.qobj
            else:
                new_op[0] = base.qobj.copy()
                for i in _set[1:]:
                    new_op[0] += self.ops[i].qobj
            new_ops.append(EvoElement(*new_op))
        self.ops = new_ops

    def compress(self):
        self.tidyup()
        sets, fsets = self._compress_make_set()
        N_sets = len(sets)
        N_fsets = len(fsets)
        num_ops = len(self.ops)

        if N_sets < num_ops and N_fsets < num_ops:
            # Both could be better
            if N_sets < N_fsets:
                self._compress_merge_qobj(sets)
            else:
                self._compress_merge_func(fsets)
            sets, fsets = self._compress_make_set()
            N_sets = len(sets)
            N_fsets = len(fsets)
            num_ops = len(self.ops)
            self._reset_type()

        if N_sets < num_ops:
            self._compress_merge_qobj(sets)
            self._reset_type()
        elif N_fsets < num_ops:
            self._compress_merge_func(fsets)
            self._reset_type()

    def permute(self, order):
        res = self.copy()
        res.cte = res.cte.permute(order)
        for op in res.ops:
            op.qobj = op.qobj.permute(order)
        return res

    # function to apply custom transformations
    def apply(self, function, *args, **kw_args):
        res = self.copy()
        cte_res = function(res.cte, *args, **kw_args)
        if not isinstance(cte_res, Qobj):
            raise TypeError("The function must return a Qobj")
        res.cte = cte_res
        for op in res.ops:
            op.qobj = function(op.qobj, *args, **kw_args)
        if self.compiled:
            self._recompile()
        return res

    def expect(self, t, state, herm=0):
        if not isinstance(t, (int, float)):
            raise TypeError("The time need to be a real scalar")
        if isinstance(state, Qobj):
            if self.cte.dims[1] == state.dims[0]:
                vec = state.full().ravel("F")
            elif self.cte.dims[1] == state.dims:
                vec = state.full().ravel("F")
            else:
                raise Exception("Dimensions do not fit")
        elif isinstance(state, np.ndarray):
            vec = state.ravel("F")
        else:
            raise TypeError("The vector must be an array or Qobj")

        if vec.shape[0] == self.cte.shape[1]:
            if self.compiled:
                exp = self.compiled_qobjevo.expect(t, vec)
            elif self.cte.issuper:
                self._dynamics_args_update(t, state)
                exp = cy_expect_rho_vec(self.__call__(t, data=True), vec, 0)
            else:
                self._dynamics_args_update(t, state)
                exp = cy_expect_psi(self.__call__(t, data=True), vec, 0)
        elif vec.shape[0] == self.cte.shape[1]**2:
            if self.compiled:
                exp = self.compiled_qobjevo.overlapse(t, vec)
            else:
                self._dynamics_args_update(t, state)
                exp = (self.__call__(t, data=True) *
                       vec.reshape((self.cte.shape[1],
                                    self.cte.shape[1])).T).trace()
        else:
            raise Exception("The shapes do not match")

        if herm:
            return exp.real
        else:
            return exp

    def mul_vec(self, t, vec):
        was_Qobj = False
        if not isinstance(t, (int, float)):
            raise TypeError("the time need to be a real scalar")
        if isinstance(vec, Qobj):
            if self.cte.dims[1] != vec.dims[0]:
                raise Exception("Dimensions do not fit")
            was_Qobj = True
            dims = vec.dims
            vec = vec.full().ravel()
        elif not isinstance(vec, np.ndarray):
            raise TypeError("The vector must be an array or Qobj")
        if vec.ndim != 1:
            raise Exception("The vector must be 1d")
        if vec.shape[0] != self.cte.shape[1]:
            raise Exception("The length do not match")

        if self.compiled:
            out = self.compiled_qobjevo.mul_vec(t, vec)
        else:
            self._dynamics_args_update(t, vec)
            out = spmv(self.__call__(t, data=True), vec)

        if was_Qobj:
            return Qobj(out, dims=dims)
        else:
            return out

    def mul_mat(self, t, mat):
        was_Qobj = False
        if not isinstance(t, (int, float)):
            raise TypeError("the time need to be a real scalar")
        if isinstance(mat, Qobj):
            if self.cte.dims[1] != mat.dims[0]:
                raise Exception("Dimensions do not fit")
            was_Qobj = True
            dims = mat.dims
            mat = mat.full()
        if not isinstance(mat, np.ndarray):
            raise TypeError("The vector must be an array or Qobj")
        if mat.ndim != 2:
            raise Exception("The matrice must be 2d")
        if mat.shape[0] != self.cte.shape[1]:
            raise Exception("The length do not match")

        if self.compiled:
            out = self.compiled_qobjevo.mul_mat(t, mat)
        else:
            self._dynamics_args_update(t, mat)
            out = self.__call__(t, data=True) * mat

        if was_Qobj:
            return Qobj(out, dims=dims)
        else:
            return out

    def compile(self, code=False, matched=False, dense=False, omp=0):
        self.tidyup()
        if self.compiled:
            return
        for _, _, op in self.dynamics_args:
            if isinstance(op, QobjEvo):
                op.compile(code, matched, dense, omp)
        if not qset.has_openmp:
            omp = 0
        if omp:
            nnz = [self.cte.data.nnz]
            for part in self.ops:
                nnz += [part.qobj.data.nnz]
            if all(qset.openmp_thresh < nz for nz in nnz):
                omp = 0
        self.omp = omp

        if matched and not self.const:
            self.compiled_qobjevo = CQobjEvoTdMatched()
            self.compiled = "matched"
        elif dense:
            self.compiled_qobjevo = CQobjEvoTdDense()
            self.compiled = "dense"
        else:
            self.compiled_qobjevo = CQobjEvoTd()
            self.compiled = "csr"

        self.compiled_qobjevo.set_data(self.cte, self.ops)
        Code = self._compile_coeff()
        self.compiled_qobjevo.set_factor(self.coeff_get)
        self.compiled_qobjevo.has_dyn_args(bool(self.dynamics_args))
        self.compiled_qobjevo.set_num_threads(self.omp)

        if code:
            return Code

    def _compile_coeff(self):
        coeffs = [op.coeff for op in self.ops]
        self._get_coeff, code = compile_united_coeff(coeffs)
        return code

    def _recompile(self):
        if self.compiled:
            self.compiled_qobjevo.set_data(self.cte, self.ops)
            self.compiled_qobjevo.has_dyn_args(bool(self.dynamics_args))
            self.compiled_qobjevo.set_num_threads(self.omp)

    def _get_coeff(self, t):
        out = []
        for part in self.ops:
            out.append(part.get_coeff(t, self.args))
        return out

    def __getstate__(self):
        _dict_ = {key: self.__dict__[key]
                  for key in self.__dict__ if key is not "compiled_qobjevo"}
        if self.compiled:
            return (_dict_, self.compiled_qobjevo.__getstate__())
        else:
            return (_dict_,)

    def __setstate__(self, state):
        self.__dict__ = state[0]
        self.compiled_qobjevo = None
        if self.compiled == "csr":
            if self.safePickle:
                # __getstate__ and __setstate__ of compiled_qobjevo pass
                # raw pointers
                # In 'safe' mod, these pointers are not used.
                # time dependence is pyfunc or cyfactor
                self.compiled_qobjevo = CQobjEvoTd()
                self.compiled_qobjevo.set_data(self.cte, self.ops)
                self.compiled_qobjevo.set_num_threads(self.omp)
                self.compiled_qobjevo.set_factor(self.coeff_get)
            else:
                self.compiled_qobjevo = CQobjEvoTd.__new__(CQobjEvoTd)
                self.compiled_qobjevo.__setstate__(state[1])

        elif self.compiled == "dense":
            CQobjEvoTdDense.__new__(CQobjEvoTdDense)
            self.compiled_qobjevo.__setstate__(state[1])

        elif self.compiled == "matched":
            self.compiled_qobjevo = \
                CQobjEvoTdMatched.__new__(CQobjEvoTdMatched)
            self.compiled_qobjevo.__setstate__(state[1])

from qutip.superoperator import vec2mat
from qutip.cy.cqobjevo import (CQobjEvoTd, CQobjEvoTdMatched, CQobjEvoTdDense)
from qutip.cy.cqobjevo_factor import (InterCoeffT, InterCoeffCte,
                                      InterpolateCoeff, StrCoeff,
                                      StepCoeffCte, StepCoeffT)
