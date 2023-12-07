__all__ = ['rhs_clear']

import os
import numpy as np
from types import FunctionType, BuiltinFunctionType
from functools import partial

from qutip.cy.codegen import Codegen
from qutip.solver import Options, config, solver_safe
from qutip.qobj import Qobj
from qutip.superoperator import spre, spost
from qutip.interpolate import Cubic_Spline


def rhs_clear():
    """
    Resets the string-format time-dependent Hamiltonian parameters.

    Parameters
    ----------

    Returns
    -------
    Nothing, just clears data from internal config module.

    """
    # time-dependent (TD) function stuff
    config.tdfunc = None     # Placeholder for TD RHS function.
    config.colspmv = None    # Placeholder for TD col-spmv function.
    config.colexpect = None  # Placeholder for TD col_expect function.
    config.string = None     # Holds string of variables to be passed to solver
    config.tdname = None     # Name of td .pyx file (used in parallel mc code)
    if "sesolve" in solver_safe:
        del solver_safe["sesolve"]
    if "mesolve" in solver_safe:
        del solver_safe["mesolve"]
    if "mcsolve" in solver_safe:
        del solver_safe["mcsolve"]


def _td_format_check(H, c_ops, solver='me'):
    """
    Checks on time-dependent format.
    """
    h_const = []
    h_func = []
    h_str = []
    h_obj = []
    # check H for incorrect format
    if isinstance(H, Qobj):
        pass
    elif isinstance(H, (FunctionType, BuiltinFunctionType, partial)):
        pass  # n_func += 1
    elif isinstance(H, list):
        for k, H_k in enumerate(H):
            if isinstance(H_k, Qobj):
                h_const.append(k)
            elif isinstance(H_k, list):
                if len(H_k) != 2 or not isinstance(H_k[0], Qobj):
                    raise TypeError("Incorrect hamiltonian specification")
                else:
                    if isinstance(H_k[1], (FunctionType,
                                           BuiltinFunctionType, partial)):
                        h_func.append(k)
                    elif isinstance(H_k[1], str):
                        h_str.append(k)
                    elif isinstance(H_k[1], Cubic_Spline):
                        h_obj.append(k)
                    elif hasattr(H_k[1], '__call__'):
                        h_obj.append(k)
                    elif isinstance(H_k[1], np.ndarray):
                        h_str.append(k)
                    else:
                        raise TypeError("Incorrect hamiltonian specification")
    else:
        raise TypeError("Incorrect hamiltonian specification")

    # the the whole thing again for c_ops
    c_const = []
    c_func = []
    c_str = []
    c_obj = []
    if isinstance(c_ops, list):
        for k in range(len(c_ops)):
            if isinstance(c_ops[k], Qobj):
                c_const.append(k)
            elif isinstance(c_ops[k], list):
                if len(c_ops[k]) != 2:
                    raise TypeError(
                        "Incorrect collapse operator specification.")
                else:
                    if isinstance(c_ops[k][1], (FunctionType,
                                                BuiltinFunctionType, partial)):
                        c_func.append(k)
                    elif isinstance(c_ops[k][1], str):
                        c_str.append(k)
                    elif isinstance(c_ops[k][1], Cubic_Spline):
                        c_obj.append(k)
                    elif hasattr(c_ops[k][1], '__call__'):
                        c_func.append(k)
                    elif isinstance(c_ops[k][1], np.ndarray):
                        c_str.append(k)
                    elif isinstance(c_ops[k][1], tuple):
                        c_str.append(k)
                    else:
                        raise TypeError(
                            "Incorrect collapse operator specification")
    else:
        raise TypeError("Incorrect collapse operator specification")

    #
    # if n_str == 0 and n_func == 0:
    #     # no time-dependence at all
    #
    if ((len(h_str) > 0 and len(h_func) > 0) or
            (len(c_str) > 0 and len(c_func) > 0)):
        raise TypeError(
            "Cannot mix string and function type time-dependence formats")

    # check to see if Cython is installed and version is high enough.
    if len(h_str) > 0 or len(c_str) > 0:
        pass
        #try:
        #    import Cython
        #except:
        #    raise Exception(
        #        "Unable to load Cython. Use Python function format.")
        #else:
        #    if Cython.__version__ < '0.21':
        #        raise Exception("Cython version (%s) is too old. Upgrade to" +
        #                        " version 0.21+" % Cython.__version__)

    # If only time-dependence is in Objects, then prefer string based format
    if (len(h_func) + len(c_func) + len(h_str) + len(c_str)) == 0:
         h_str += h_obj #Does nothing if not objects
         c_str += c_obj
    else:
        # Combine Hamiltonian objects
        if len(h_func) > 0:
            h_func += h_obj
        elif len(h_str) > 0:
            h_str += h_obj

        #Combine collapse objects
        if len(c_func) > 0:
            c_func += c_obj
        elif len(c_str) > 0:
            c_str += c_obj

    if solver == 'me':
        return (len(h_const + c_const),
                len(h_func) + len(c_func),
                len(h_str) + len(c_str))

    elif solver == 'mc':

        #   H      C_ops    #
        #   --     -----    --
        #   NO      NO      00
        #   NO     STR      01
        #   NO     FUNC     02
        #
        #   STR    NO       10
        #   STR    STR      11
        #
        #   FUNC   NO       20
        #
        #   FUNC   FUNC     22

        if isinstance(H, FunctionType):
            time_type = 3
        # Time-indepdent problems
        elif ((len(h_func) == len(h_str) == 0) and
                (len(c_func) == len(c_str) == 0)):
            time_type = 0

        # constant Hamiltonian, time-dependent collapse operators
        elif len(h_func) == len(h_str) == 0:
            if len(c_str) > 0:
                time_type = 1
            elif len(c_func) > 0:
                time_type = 2
            else:
                raise Exception("Error determining time-dependence.")

        # list style Hamiltonian
        elif len(h_str) > 0:
            if len(c_func) == len(c_str) == 0:
                time_type = 10
            elif len(c_str) > 0:
                time_type = 11
            else:
                raise Exception("Error determining time-dependence.")

        # Python function style Hamiltonian
        elif len(h_func) > 0:
            if len(c_func) == len(c_str) == 0:
                time_type = 20
            elif len(c_func) > 0:
                time_type = 22
            else:
                raise Exception("Error determining time-dependence.")

        return time_type, [h_const, h_func, h_str], [c_const, c_func, c_str]


def _td_wrap_array_str(H, c_ops, args, times):
    """
    Wrap numpy-array based time-dependence in the string-based time dependence
    format
    """
    n = 0
    H_new = []
    c_ops_new = []
    args_new = {}

    if not isinstance(H, list):
        H_new = H
    else:
        for Hk in H:
            if isinstance(Hk, list) and isinstance(Hk[1], np.ndarray):
                H_op, H_td = Hk
                td_array_name = "_td_array_%d" % n
                H_td_str = '(0 if (t > %f) else %s[int(round(%d * (t/%f)))])' %\
                    (times[-1], td_array_name, len(times) - 1, times[-1])
                args_new[td_array_name] = H_td
                H_new.append([H_op, H_td_str])
                n += 1
            else:
                H_new.append(Hk)

    if not isinstance(c_ops, list):
        c_ops_new = c_ops
    else:
        for ck in c_ops:
            if isinstance(ck, list) and isinstance(ck[1], np.ndarray):
                c_op, c_td = ck
                td_array_name = "_td_array_%d" % n
                c_td_str = '(0 if (t > %f) else %s[int(round(%d * (t/%f)))])' %\
                    (times[-1], td_array_name, len(times) - 1, times[-1])
                args_new[td_array_name] = c_td
                c_ops_new.append([c_op, c_td_str])
                n += 1
            else:
                c_ops_new.append(ck)

    if not args_new:
        args_new = args
    elif isinstance(args, dict):
        args_new.update(args)
    else:
        raise ValueError("Time-dependent array format requires args to " +
                         "be a dictionary")

    return H_new, c_ops_new, args_new
