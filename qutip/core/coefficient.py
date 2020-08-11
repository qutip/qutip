import numpy as np
import scipy
import os
import sys
import re
import dis
import hashlib
import glob
import importlib
import shutil
import numbers
from collections import defaultdict
from .. import settings as qset
from .data import Data
from .interpolate import Cubic_Spline
from .cy.coefficient import (InterpolateCoefficient, InterCoefficient,
                             StepCoefficient, FunctionCoefficient,
                             SumCoefficient, MulCoefficient,
                             ConjCoefficient, NormCoefficient,
                             ShiftCoefficient, StrFunctionCoefficient,
                             Coefficient)
from setuptools import setup, Extension
try:
    from Cython.Build import cythonize
except ImportError:
    pass
from warnings import warn


__all__ = ["coefficient", "CompilationOptions"]


class StringParsingWarning(Warning):
    pass


def coefficient(base, *, tlist=None, args={}, args_ctypes={},
                _stepInterpolation=False, compile_opt=None):
    """Coefficient for Qutip time dependent systems.
    The coefficients are either a function, a string or a numpy array.

    For function format, the function signature must be f(t, args).
    *Examples*
        def f1_t(t, args):
            return np.exp(-1j * t * args["w1"])

        coeff = coefficient(f1_t, args={"w1":1.})

    For string based coeffients, the string must be a compilable python code
    resulting in a complex. The following symbols are defined:
        sin cos tan asin acos atan pi
        sinh cosh tanh asinh acosh atanh
        exp log log10 erf zerf sqrt
        real imag conj abs norm arg proj
        numpy as np, and scipy.special as spe.
    *Examples*
        coeff = coefficient('exp(-1j*w1*t)', args={"w1":1.})
    'args' is needed for string coefficient at compilation.
    It is a dict of (name:object). The keys must be a valid variables string.

    Compilation options can be passed as "compile_opt=CompilationOptions(...)".

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
        H = QobjEvo(np.exp(-1j*tlist), tlist=tlist)
    """
    if isinstance(base, Coefficient):
        return base

    if isinstance(base, Cubic_Spline):
        return InterpolateCoefficient(base)

    elif isinstance(base, np.ndarray):
        if len(base.shape) != 1:
            raise ValueError("The array to interpolate must be a 1D array")
        if base.shape != tlist.shape:
            raise ValueError("tlist must be the same len "
                             "as the array to interpolate")
        base = base.astype(np.complex128)
        tlist = tlist.astype(np.float64)
        if not _stepInterpolation:
            return InterCoefficient(base, tlist)
        else:
            return StepCoefficient(base, tlist)

    elif isinstance(base, str):
        if compile_opt is None:
            compile_opt = CompilationOptions()
        return coeff_from_str(base, args, args_ctypes, compile_opt)

    elif callable(base):
        # TODO add tests?
        return FunctionCoefficient(base, args)
    else:
        raise ValueError("coefficient format not understood")


def norm(coeff):
    """ return a Coefficient with is the norm: |c|^2.
    """
    return NormCoefficient(coeff)


def conj(coeff):
    """ return a Coefficient with is the conjugate.
    """
    return ConjCoefficient(coeff)


def shift(coeff, _t0=0):
    """ return a Coefficient in which t is shifted by _t0.
    """
    return ShiftCoefficient(coeff, _t0)


def reduce(coeff, args):
    """ Reduce decorated string coefficient to 1 object:
    c = coefficient("t")
    optimize(c + conj(c)) => coefficient("t+conj(t)")
    """
    reduced = coeff.optstr()
    if reduced:
        return coefficient(reduced, args=args)
    else:
        return coeff


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%      Everything under this is for string compilation      %%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class CompilationOptions:
    """
    Options for compilation.
    use_cython: bool
        execute strings as python code instead of cython.

    Type management options.
    accept_int : None, bool
        Whether integer constants and args are kept or upgraded to float
        If `None`, use in if array subscrition is used.
    accept_float : bool
        Whether float are kept as float or upgraded to complex.
    no_types : bool
        Give up on detecting and using c types.
    recompile : bool
        Do not use previously made files but build a new one.
    compiler_flags : str
        Flags to pass to the compiler, ex: "-Wall -O3"...
        Flags not matching your comiler and OS may cause compilation to fail.
        Use "recompile=True", when trying to if the string pattern was
        previously used.
    link_flags : str
        Libraries to link to pass to the compiler. They can not be used to add
        function to the string coefficient.
    extra_import : str
        Cython code to add at the head of the file. Can be used to add extra
        import or import c code etc. ex:
        "from scipy.linalg import det"
        "from qutip.core.data import CSR"
    """
    # TODO: use the Options decorator when merged and put in Core options (v5)

    # use cython for compiling string coefficient.
    try:
        import cython
        use_cython = True
    except ImportError:
        use_cython = False
    # In compiled Coefficient, are int kept as int?
    # None indicate to look for list subscription
    accept_int = None
    # In compiled Coefficient, are float considered as complex?
    accept_float = True
    # In compiled Coefficient, is static typing used?
    # Result is faster, but can cause errors if subscription
    # (a[1], b["a"]) if used.
    no_types = False
    # Skip saved previously compiled files and force compilation
    recompile = False
    # Compilation flags and link flags to pass to the compiler
    link_flags = ""
    compiler_flags = ""
    if (sys.platform == 'win32' and os.environ.get('MSYSTEM') is None):
        compiler_flags = '/w /Ox'
    elif sys.platform == 'darwin':
        compiler_flags = '-w -O3 -funroll-loops -mmacosx-version-min=10.9'
        link_flags += '-mmacosx-version-min=10.9'
    else:
        compiler_flags = '-w -O3 -funroll-loops'
    # Extra_header
    extra_import = ""

    def __init__(self,
                 use_cython=None,
                 accept_int=None,
                 accept_float=None,
                 no_types=None,
                 recompile=None,
                 compiler_flags=None,
                 link_flags=None,
                 extra_import=None):
        if use_cython is not None:
            self.use_cython = use_cython
        if accept_int is not None:
            self.accept_int = accept_int
        if accept_float is not None:
            self.accept_float = accept_float
        if no_types is not None:
            self.no_types = no_types
        if recompile is not None:
            self.recompile = recompile
        if compiler_flags is not None:
            self.compiler_flags = compiler_flags
        if link_flags is not None:
            self.link_flags = link_flags
        if extra_import is not None:
            self.extra_import = extra_import


def proj(x):
    if np.isfinite(x):
        return (x)
    else:
        return np.inf + 0j * np.imag(x)


str_env = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "pi": np.pi,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "asinh": np.arcsinh,
    "acosh": np.arccosh,
    "atanh": np.arctanh,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "erf": scipy.special.erf,
    "zerf": scipy.special.erf,
    "sqrt": np.sqrt,
    "real": np.real,
    "imag": np.imag,
    "conj": np.conj,
    "abs": np.abs,
    "norm": lambda x: np.abs(x)**2,
    "arg": np.angle,
    "proj": proj,
    "np": np,
    "spe": scipy.special}


def coeff_from_str(base, args, args_ctypes, compile_opt):
    """
    Entry point for string based coefficients
    - Test if the string is valid
    - Parse: "cos(a*t)" and "cos( w1 * t )"
        should be recognised as the same compiled object.
    - Verify if already compiled and compile if not
    """
    # First, a sanity check before thinking of compiling
    if not compile_opt.extra_import:
        try:
            env = {"t": 0}
            env.update(args)
            exec(base, str_env, env)
        except Exception as err:
            raise Exception("Invalid string coefficient") from err
    # Do we even compile?
    if not qset.use_cython or not compile_opt.use_cython:
        return StrFunctionCoefficient(base, args)
    # Parsing tries to make the code in common pattern
    parsed, variables, constants, raw = try_parse(base, args,
                                                  args_ctypes, compile_opt)
    # Once parsed, the code should be unique enough to get a filename
    hash_ = hashlib.sha256(bytes(parsed, encoding='utf8'))
    file_name = "qtcoeff_" + hash_.hexdigest()[:30]
    # See if it already exist, if not write and cythonize it
    coeff = try_import(file_name, parsed)
    if coeff is None or compile_opt.recompile:
        code = make_cy_code(parsed, variables, constants, raw, compile_opt)
        coeff = compile_code(code, file_name, parsed, compile_opt)
    keys = [key for _, key, _ in variables]
    const = [fromstr(val) for _, val, _ in constants]
    return coeff(base, keys, const, args)


def try_import(file_name, parsed_in):
    """ Import the compiled coefficient if existing and check for
    name collision.
    """
    coeff = None
    try:
        mod = importlib.import_module(file_name)
        coeff = getattr(mod, "StrCoefficient")
        parsed_saved = getattr(mod, "parsed_code")
    except Exception as e:
        parsed_saved = ""
    if parsed_saved and parsed_in != parsed_saved:
        # hash collision!
        coeff = None
    return coeff


def make_cy_code(code, variables, constants, raw, compile_opt):
    """
    Generate the code for the string coefficients.
    """
    cdef_cte = ""
    init_cte = ""
    for i, (name, val, ctype) in enumerate(constants):
        cdef_cte += "        {} {}\n".format(ctype, name[5:])
        init_cte += "        {} = cte[{}]\n".format(name, i)
    cdef_var = ""
    init_var = ""
    args_var = ""
    call_var = ""
    for i, (name, val, ctype) in enumerate(variables):
        cdef_var += "        str key{}\n".format(i)
        cdef_var += "        {} {}\n".format(ctype, name[5:])
        init_var += "        self.key{} = var[{}]\n".format(i, i)
        args_var += "        {} = args[self.key{}]\n".format(name, i)
        if raw:
            call_var += "        cdef {} {} = {}\n".format(ctype, val, name)

    code = """#cython: language_level=3
# This file is generated automatically by QuTiP.

import numpy as np
import scipy.special as spe
cimport cython
from qutip.core.cy.coefficient cimport Coefficient
from qutip.core.cy.math cimport erf, zerf
from qutip.core.cy.complex_math cimport *
from qutip.core.data cimport Data
cdef double pi = 3.14159265358979323
{}

parsed_code = "{}"


@cython.auto_pickle(True)
cdef class StrCoefficient(Coefficient):
    cdef:
        str codeString
{}{}

    def __init__(self, base, var, cte, args):
        self.codeString = base
{}{}        self.arguments(args)

    cpdef void arguments(self, dict args) except *:
{}        pass

    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cdef complex _call(self, double t) except *:
{}        return {}

    def optstr(self):
        return self.codeString
""".format(compile_opt.extra_import, code, cdef_cte, cdef_var,
           init_cte, init_var, args_var, call_var, code)
    return code


def compile_code(code, file_name, parsed, c_opt):
    root = qset.tmproot
    full_file_name = os.path.join(root, file_name)
    file_ = open(full_file_name + ".pyx", "w")
    file_.writelines(code)
    file_.close()
    oldargs = sys.argv
    try:
        sys.argv = ["setup.py", "build_ext", "--inplace"]
        coeff_file = Extension(file_name,
                               sources=[full_file_name + ".pyx"],
                               extra_compile_args=c_opt.compiler_flags.split(),
                               extra_link_args=c_opt.link_flags.split(),
                               include_dirs=[np.get_include()],
                               language='c++')
        setup(ext_modules=cythonize(coeff_file, force=c_opt.recompile))
    except Exception as e:
        raise Exception("Could not compile") from e
    try:
        libfile = glob.glob(file_name + "*")[0]
        shutil.move(libfile, os.path.join(root, libfile))
    except Exception:
        warn("File")
    sys.argv = oldargs
    return try_import(file_name, parsed)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%        Everything under this is for parsing string        %%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Parsing here is extracting constants and args name to replace them with
# attribute of the Coefficient so similar string like:
# "2.*cos(a*t)", "5.2 * cos(w1 *t)", "5 * cos(w3 * t)"
# are all reconized as the same compiled object and only compiled once.
# Weakness:
#   typing: "1" and "1j" or the type of args ("w1") make different object
#   complex: "1+1j" is seens as cte(double) + cte(complex)
#   negative: "-1" is not seens as a constant but "- constant"
#
# int and double can be seens as complex with flags in CompilationOptions

def fromstr(base):
    """Read a varibles in a string"""
    ls = {}
    exec("out = " + base, {}, ls)
    return ls["out"]


typeCodes = {
    "Data": "_datalayer",
    "complex": "_cpl",
    "double": "_dbl",
    "int": "_int",
    "str": "_str",
    "object": "_obj"
}


def compileType(value):
    """Obtain the index of typeCodes that correspond to the value
    4.5 -> 'double'..."""
    if isinstance(value, Data):
        ctype = "Data"
    elif isinstance(value, numbers.Integral):
        ctype = "int"
    elif isinstance(value, numbers.Real):
        ctype = "double"
    elif isinstance(value, numbers.Complex):
        ctype = "complex"
    elif isinstance(value, str):
        ctype = "str"
    else:
        ctype = "object"
    return ctype


def find_type_from_str(chars):
    """ '1j' -> complex """
    try:
        lc = {}
        exec("out = " + chars, {}, lc)
        return compileType(lc["out"])
    except Exception:
        return None


def fix_type(ctype, accept_int, accept_float):
    """int and double could be complex to limit the number of compiled object.
    change the types is we choose not to support all.
    """
    if ctype == "int" and not accept_int:
        ctype = "double"
    if ctype == "double" and not accept_float:
        ctype = "complex"
    return ctype


def extract_constant(code):
    """Look for floating and complex constants and replace them with variable.
    """
    code = " " + code + " "
    contants = []
    code = extract_cte_pattern(code, contants,
                               "[^0-9a-zA-Z_][0-9]*[.]?[0-9]+e[+-]?[0-9]*[j]?")
    code = extract_cte_pattern(code, contants,
                               "[^0-9a-zA-Z_][0-9]+[.]?[0-9]*e[+-]?[0-9]*[j]?")
    code = extract_cte_pattern(code, contants,
                               "[^0-9a-zA-Z_][0-9]+[.]?[0-9]*[j]?")
    code = extract_cte_pattern(code, contants,
                               "[^0-9a-zA-Z_][0-9]*[.]?[0-9]+[j]?")
    return code, contants


def extract_cte_pattern(code, constants, pattern):
    """replace the constant following a pattern with variable"""
    const_strs = re.findall(pattern, code)
    for cte in const_strs:
        name = " _cte_temp{}_ ".format(len(constants))
        code = code.replace(cte, cte[0] + name, 1)
        constants.append((name[1:-1], cte[1:], find_type_from_str(cte[1:])))
    return code


def space_parts(code, names):
    """Force spacing: single space between element"""
    for name in names:
        code = re.sub("(?<=[^0-9a-zA-Z_])" + name + "(?=[^0-9a-zA-Z_])",
                      " " + name + " ", code)
    code = " ".join(code.split())
    return code


def parse(code, args, compile_opt):
    """
    Read the code and rewrite it in a reutilisable form:
    Ins:
        '2.*cos(a*t)', {"a":5+1j}
    Outs:
        code = 'self._cte_dbl0 * cos ( self._arg_cpl0 * t )'
        variables = [('self._arg_cpl0', 'a', 'complex')]
        ordered_constants = [('self._cte_dbl0', 2, 'double')]
    """
    code, constants = extract_constant(code)
    names = re.findall("[0-9a-zA-Z_]+", code)
    code = space_parts(code, names)
    constants_names = [const[0] for const in constants]
    new_code = []
    ordered_constants = []
    variables = []
    typeCounts = defaultdict(lambda: 0)
    accept_int = compile_opt.accept_int
    accept_float = compile_opt.accept_float
    if accept_int is None:
        # If there is a subscript: a[b] int are always accepted to be safe
        # with TypeError
        accept_int = "SUBSCR" in dis.Bytecode(code).dis()
    for word in code.split():
        if word not in names:
            # syntax
            new_code.append(word)
        elif word in args:
            # find first if the variable is use more than once and reuse
            var_name = [var_name for var_name, name, _ in variables
                        if word == name]
            if var_name:
                var_name = var_name[0]
            else:
                ctype = compileType(args[word])
                ctype = fix_type(ctype, accept_int, accept_float)
                var_name = ("self._arg" + typeCodes[ctype] +
                            str(typeCounts[ctype]))
                typeCounts[ctype] += 1
                variables.append((var_name, word, ctype))
            new_code.append(var_name)
        elif word in constants_names:
            name, val, ctype = constants[int(word[9:-1])]
            ctype = fix_type(ctype, accept_int, accept_float)
            cte_name = "self._cte" + typeCodes[ctype] +\
                       str(len(ordered_constants))
            new_code.append(cte_name)
            ordered_constants.append((cte_name, val, ctype))
        else:
            # Hopefully a buildin or known object
            new_code.append(word)
        code = " ".join(new_code)
    return code, variables, ordered_constants


def use_hinted_type(var_tuple, ncode, args_ctypes):
    name, key, type_ = var_tuple
    if key in args_ctypes:
        code = code.replace(cte, cte[0] + name, 1)


def try_parse(code, args, args_ctypes, compile_opt):
    """
    Try to parse and verify that the result is still usable.
    """
    ncode, variables, constants = parse(code, args, compile_opt)
    if compile_opt.no_types:
        # Fallback to all object
        variables = [(f, s, "object") for f, s, _ in variables]
        constants = [(f, s, "object") for f, s, _ in constants]
    variables_manually_typed = []
    for i, (name, key, type_) in enumerate(variables):
        if key in args_ctypes:
            new_name = "self._custom_" + args_ctypes[key] + str(i)
            ncode = ncode.replace(name, new_name)
            variables_manually_typed.append((new_name, key, args_ctypes[key]))
        else:
            variables_manually_typed.append((name, key, type_))
    variables = variables_manually_typed
    if (compile_opt.extra_import or
        test_parsed(ncode, variables, constants, args)):
            return ncode, variables, constants, False
    else:
        warn("Could not find c types", StringParsingWarning)
        remaped_variable = []
        for _, name, ctype in variables:
            remaped_variable.append(("self." + name, name, "object"))
        return code, remaped_variable, [], True


def test_parsed(code, variables, constants, args):
    """
    Test if parsed code broke anything.
    """
    class DummySelf:
        pass
    [setattr(DummySelf, cte[0][5:], fromstr(cte[1])) for cte in constants]
    [setattr(DummySelf, var[0][5:], args[var[1]]) for var in variables]
    loc_env = {"t": 0, 'self': DummySelf}
    try:
        exec(code, str_env, loc_env)
    except Exception as e:
        return False
    return True
