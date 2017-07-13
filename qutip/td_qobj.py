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

from qutip import Qobj
from qutip.interpolate import Cubic_Spline
from functools import partial
from types import FunctionType, BuiltinFunctionType
import numpy as np
from numbers import Number
from qutip.superoperator import liouvillian


class td_Qobj:
    """A class for representing time-dependent quantum objects,
    such as quantum operators and states.

    The td_Qobj class is a representation of time-dependent Qutip quantum
    objects (Qobj). This class implements math operations :
        +,- : td_Qobj, Qobj
        * : Qobj, C-number
        / : C-number
    and some common linear operator/state operations. The td_Qobj
    are constructed from a nested list of Qobj with their time-dependent
    coefficients.

    *Examples*

        H = [H0, [H1, 'sin(w*t)'], [H2, f1_t], [H3, np.sin(3*w*tlist)]]

        where f1_t ia a python functions with signature f_t(t, args, **kw_args).

    Parameters
    ----------
    Q_object : array_like
        Data for vector/matrix representation of the quantum object.
    args : dictionary that contain the arguments for
    tlist : array_like
        List of times at which the numpy-array coefficients are applied. Times
        must be equidistant and start from 0.
    #copy : bool
    #    Flag specifying whether Qobj should get a copy of the
    #    input data, or use the original.
    #check : bool
    #    Check if the supplied Qobj are compatible


    Attributes
    ----------
    cte : Qobj
        Constant part of the td_Qobj
    args : map
        arguments of the coefficients
    const : bool
        Indicates if quantum object is Constant
    tlist : array_like
        List of times at which the numpy-array coefficients are applied.
    ops : list
        List of Qobj and the coefficients.
        [(Qobj, coefficient as a function, original coefficient, type), ... ]
        type :
            1: function
            2: string
            3: np.array
    op_call : function
        User defined function as a td_Qobj. For wrapping function Hamiltonian
        given to solvers.


    To implements: callback to the self.cte
        dims : list
            List of dimensions keeping track of the tensor structure.
        shape : list
            Shape of the underlying `data` array.
        type : str
            Type of quantum object: 'bra', 'ket', 'oper', 'operator-ket',
            'operator-bra', or 'super'.
        superrep : str
            Representation used if `type` is 'super'. One of 'super'
            (Liouville form) or 'choi' (Choi matrix with tr = dimension).
        isherm : bool
            Indicates if quantum object represents Hermitian operator.
        iscp : bool
            Indicates if the quantum object represents a map, and if that map is
            completely positive (CP).
        ishp : bool
            Indicates if the quantum object represents a map, and if that map is
            hermicity preserving (HP).
        istp : bool
            Indicates if the quantum object represents a map, and if that map is
            trace preserving (TP).
        iscptp : bool
            Indicates if the quantum object represents a map that is completely
            positive and trace preserving (CPTP).
            Indicates if the quantum object represents a map that is completely
            positive and trace preserving (CPTP).
        isket : bool
            Indicates if the quantum object represents a ket.
        isbra : bool
            Indicates if the quantum object represents a bra.
        isoper : bool
            Indicates if the quantum object represents an operator.
        issuper : boolt
            Indicates if the quantum object represents a superoperator.
        isoperket : bool
            Indicates if the quantum object represents an operator in column
            vector form.
        isoperbra : bool
            Indicates if the quantum object represents an operator in row vector
            form.

    Methods
    -------
    apply(f, *args, **kw_args)
        Apply the function f to every Qobj. f(Qobj) -> Qobj
        Return a modified td_Qobj and let the original one untouched
    copy()
        Create copy of Qobj
    conj()
        Conjugate of quantum object.
    dag()
        Adjoint (dagger) of quantum object.
    permute(order)
        Returns composite qobj with indices reordered.
    ptrace(sel)
        Returns quantum object for selected dimensions after performing
        partial trace.
    tidyup(atol=1e-12)
        Removes small elements from quantum object.
    trans()
        Transpose of quantum object.
    """

    def __init__(self, Q_object=[], args={}, tlist=None):
        self.const = False
        self.args = args
        self.cte = None
        self.tlist = tlist
        self.op_call = None

        op_type = self._td_format_check_single(Q_object, tlist)
        self.op_type = op_type
        self.ops = []

        if isinstance(op_type, int):
            if op_type == 0:
                self.cte = Q_object
                self.const = True
                if Q_object.issuper:
                    self.issuper = True
            elif op_type == 1:
                # a function, no test to see if the
                # function does return a Qobj.
                self.op_call = Q_object
            elif op_type == -1:
                pass

        else:
            compile_list = []
            compile_count = 0
            for type_, op in zip(op_type, Q_object):
                if type_ == 0:
                    if self.cte is None:
                        self.cte = op
                    else:
                        self.cte += op
                elif type_ == 1:
                    self.ops.append([op[0], op[1], op[1], 1])
                elif type_ == 2:
                    self.ops.append([op[0], None, op[1], 2])
                    compile_list.append((op[1], compile_count))
                    compile_count += 1
                elif type_ == 3:
                    l = len(self.ops)
                    self.ops.append([op[0], None,
                                     op[1].copy(), 3])

                    self.ops[-1][1] = lambda t, *args, l=l, **kw_args:
                                    (0 if (t > self.tlist[-1])
                                    else self.ops[l][2][int(
                                    round((len(self.tlist) - 1) *
                                    (t/self.tlist[-1])))])
                else:
                    raise Exception("Should never be here")

            if compile_count:
                str_funcs = self._compile_str_single(compile_list)
                count = 0
                for op in self.ops:
                    if op[3] == 2:
                        op[1] = str_funcs[count]
                        count += 1

            if not self.cte:
                self.cte = self.ops[0][0]
                for op in self.ops[1:]:
                    self.cte += op[0]
                self.cte *= 0.

            if not self.ops:
                self.const = True

    # Different function to get the state
    def __call__(self, t):
        if self.op_call is not None:
            return self.op_call(t, **self.args)
        op_t = self.cte
        for part in self.ops:
            op_t += part[0] * part[1](t, **self.args)
        return op_t

    def _td_array_to_str(self, op_np2, times):
        """
        Wrap numpy-array based time-dependence in the string-based
        time dependence format
        """
        n = 0
        str_op = []
        np_args = {}

        for op in op_np2:
            td_array_name = "_td_array_%d" % n
            H_td_str = '(0 if (t > %f) else %s[int(round(%d * (t/%f)))])' %\
                (times[-1], td_array_name, len(times) - 1, times[-1])
            np_args[td_array_name] = op[1]
            str_op.append([op[0], H_td_str])
            n += 1

        return str_op, np_args

    def _td_format_check_single(self, Q_object, tlist=None):
        op_type = []

        if isinstance(Q_object, Qobj):
            op_type = 0
        elif isinstance(Q_object, (FunctionType,
                                   BuiltinFunctionType, partial)):
            op_type = 1
        elif isinstance(Q_object, list):
            if (len(Q_object) == 0):
                op_type = -1
            for op_k in Q_object:
                if isinstance(op_k, Qobj):
                    op_type.append(0)
                elif isinstance(op_k, list):
                    if not isinstance(op_k[0], Qobj):
                        raise TypeError("Incorrect Q_object specification")
                    elif len(op_k) == 2:
                        if isinstance(op_k[1], (FunctionType,
                                                BuiltinFunctionType, partial)):
                            op_type.append(1)
                        elif isinstance(op_k[1], str):
                            op_type.append(2)
                        elif isinstance(op_k[1], np.ndarray):
                            if not isinstance(tlist, np.ndarray) or not \
                                        len(op_k[1]) == len(tlist):
                                raise TypeError("Time list do not match")
                            op_type.append(3)
                        elif isinstance(op_k[1], Cubic_Spline):
                            raise NotImplementedError(
                                    "Cubic_Spline not supported")
                        #    h_obj.append(k)
                        else:
                            raise TypeError("Incorrect Q_object specification")
                    else:
                        raise TypeError("Incorrect Q_object specification")
        else:
            raise TypeError("Incorrect Q_object specification")
        return op_type

    def _generate_op(self):
        compiled_str_coeff = self._compile_str_()
        if(len(self.args) == 0):
            def str_func_with_np(t):
                return compiled_str_coeff(t)
        else:
            def str_func_with_np(t):
                return compiled_str_coeff(t, *(list(zip(*self.args))[1]))

        return compiled_str_coeff

    def _compile_str_single(self, compile_list):

        import os
        _cython_path = os.path.dirname(os.path.abspath(__file__)).replace(
                    "\\", "/")
        _include_string = "'"+_cython_path + "/cy/complex_math.pxi'"

        all_str = ""
        for op in compile_list:
            all_str += op[0]

        filename = "td_Qobj_"+str(hash(all_str))[1:]

        Code = """
# This file is generated automatically by QuTiP.

import numpy as np
cimport numpy as np
cimport cython
np.import_array()
cdef extern from "numpy/arrayobject.h" nogil:
    void PyDataMem_NEW_ZEROED(size_t size, size_t elsize)
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
from qutip.cy.spmatfuncs cimport spmvpy
from qutip.cy.interpolate cimport interp, zinterp
from qutip.cy.math cimport erf
cdef double pi = 3.14159265358979323

include """+_include_string+"\n"

        for str_coeff in compile_list:
            Code += self._str_2_code(str_coeff)

        file = open(filename+".pyx", "w")
        file.writelines(Code)
        file.close()

        str_func = []
        imp = ' import '
        self.str_func = []
        for i in range(len(compile_list)):
            func_name = '_str_factor_' + str(i)
            import_code = compile('from ' + filename + ' import ' + func_name +
                                  "\nstr_func.append(" + func_name + ")",
                                  '<string>', 'exec')
            exec(import_code, locals())

        try:
            os.remove(filename+".pyx")
        except:
            pass

        return str_func

    def _str_2_code(self, str_coeff):

        func_name = '_str_factor_' + str(str_coeff[1])

        Code = """

@cython.boundscheck(False)
@cython.wraparound(False)

def """ + func_name + "(double t"
        Code += self._get_arg_str()
        Code += "):\n"
        Code += "    return " + str_coeff[0] + "\n"

        return Code

    def _get_arg_str(self):
        if len(self.args) == 0:
            return ''

        ret = ''
        for name, value in self.args.items():
            if isinstance(value, np.ndarray):
                ret += ",\n        np.ndarray[np.%s_t, ndim=1] %s" % \
                    (value.dtype.name, name)
            else:
                if isinstance(value, (int, np.int32, np.int64)):
                    kind = 'int'
                elif isinstance(value, (float, np.float32, np.float64)):
                    kind = 'float'
                elif isinstance(value, (complex, np.complex128)):
                    kind = 'complex'
                ret += ",\n        " + kind + " " + name
        return ret

    def copy(self):
        new = td_Qobj(self.cte.copy())
        new.const = self.const
        new.op_call = self.op_call
        new.args = self.args
        for op in self.ops:
            new.ops.append([None, None, None, None])
            new.ops[-1][0] = op[0].copy()
            new.ops[-1][3] = op[3]
            new.ops[-1][2] = op[2]
            if new.ops[-1][3] in [1, 2]:
                new.ops[-1][1] = op[1]
            elif new.ops[-1][3] == 3:
                l = len(new.ops)-1
                new.ops[l][1] = lambda t, *args, l=l, **kw_args:
                                (0 if (t > self.tlist[-1])
                                else self.ops[l][2][
                                int(round((len(self.tlist) - 1) *
                                (t/self.tlist[-1])))])
        return new

    def __add__(self, other):
        res = self.copy()
        res += other
        return res

    def __radd__(self, other):
        res = self.copy()
        res += other
        return res

    def __iadd__(self, other):
        if isinstance(other, td_Qobj):
            self.cte += other.cte
            self.ops += other.ops
            self.args = {**self.args, **other.args}
            self.const = self.const and other.const
            if self.tlist is None:
                self.tlist = other.tlist
            else:
                if other.tlist is None:
                    pass
                elif len(other.tlist) != len(self.tlist) or \
                        other.tlist[-1] != self.tlist[-1]:
                    raise Exception("tlist are not compatible")
        else:
            self.cte += other
        return self

    def __sub__(self, other):
        res = self.copy()
        res -= other
        return res

    def __rsub__(self, other):
        res = self.copy()
        res -= other
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
        res *= other
        return res

    def __imul__(self, other):
        if isinstance(other, Qobj) or isinstance(other, Number):
            self.cte *= other
            for op in enumerate(ops):
                op[0] *= other
            return res
        else:
            raise TypeError("td_qobj can only be multiplied" +
                            " with Qobj or numbers")
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
        res.cte = (-res.cte)
        for op in res.ops:
            op[0] = -op[0]
        return res

    def trans(self):
        res = self.copy()
        res.cte = res.cte.trans()
        for op in res.ops:
            op[0] = op[0].trans()
        return res

    def conj(self):
        res = self.copy()
        res.cte = res.cte.conj()
        for op in res.ops:
            op[0] = op[0].conj()
        return res

    def dag(self):
        res = self.copy()
        res.cte = res.cte.dag()
        for op in res.ops:
            op[0] = op[0].dag()
        return res

    def apply(self, function, *args, **kw_args):
        res = self.copy()
        cte_res = function(res.cte, *args, **kw_args)
        if not isinstance(cte_res, Qobj):
            raise TypeError("The function must return a Qobj")
        res.cte = cte_res
        for op in res.ops:
            op[0] = function(op[0], *args, **kw_args)
        return res

    def tidyup(self, atol=1e-12):
        self.cte = self.cte.tidyup(atol)
        for op in res.ops:
            op[0] = op[0].tidyup(atol)
        return res

    def permute(self, order):
        res = self.copy()
        res.cte = res.cte.permute(order)
        for op in res.ops:
            op[0] = op[0].permute(order)
        return res

    def ptrace(self, sel):
        res = self.copy()
        res.cte = res.cte.ptrace(sel)
        for op in res.ops:
            op[0] = op[0].ptrace(sel)
        return res


def td_liouvillian(H, c_ops=[], chi=None):
    """Assembles the Liouvillian superoperator from a Hamiltonian
    and a ``list`` of collapse operators. Accept time dependant
    operator and return a td_qobj

    Parameters
    ----------
    H : qobj
        System Hamiltonian.

    c_ops : array_like
        A ``list`` or ``array`` of collapse operators.

    Returns
    -------
    L : td_qobj
        Liouvillian superoperator.

    """
    L = None

    if chi and len(chi) != len(c_ops):
        raise ValueError('chi must be a list with same length as c_ops')

    if H is not None:
        if not isinstance(H, td_Qobj):
            L = td_Qobj(H)
        else:
            L = H
        L = L.apply(liouvillian, chi=chi)

    if isinstance(c_ops, list) and len(c_ops) > 0:
        def liouvillian_c(c_ops, chi):
            return liouvillian(None, c_ops=[c_ops], chi=chi)
        for c in c_ops:
            if not isinstance(c, td_Qobj):
                cL = td_Qobj(c)
            else:
                cL = c

            if L is None:
                L = cL.apply(liouvillian_c, chi=chi)
            else:
                L += cL.apply(liouvillian_c, chi=chi)

    return L
