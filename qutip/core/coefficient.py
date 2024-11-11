# Required for Sphinx to follow autodoc_type_aliases
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
import scipy
import scipy.interpolate
import os
import sys
import re
import dis
import hashlib
import glob
import importlib
import warnings
import numbers
from collections import defaultdict
try:
    from setuptools import setup, Extension
    from Cython.Build import cythonize
    import filelock
except ImportError:
    pass

from ..settings import settings as qset
from .options import QutipOptions
from .data import Data
from .cy.coefficient import (
    Coefficient, InterCoefficient, FunctionCoefficient, StrFunctionCoefficient,
    ConjCoefficient, NormCoefficient, ConstantCoefficient
)
from qutip.typing import CoefficientLike


__all__ = ["coefficient", "CompilationOptions", "Coefficient",
           "clean_compiled_coefficient"]


class StringParsingWarning(Warning):
    pass


def _return(base, **kwargs):
    return base


# The `coefficient` function is dispatcher for the type of the `base` to the
# function that created the `Coefficient` object. `coefficient_builders` stores
# the map `type -> function(base, **kw)`. Optional module can add their
# `Coefficient` specializations here.
coefficient_builders = {
    Coefficient: _return,
    np.ndarray: InterCoefficient,
    scipy.interpolate.PPoly: InterCoefficient.from_PPoly,
    scipy.interpolate.BSpline: InterCoefficient.from_Bspline,
    numbers.Number: ConstantCoefficient,
}


def coefficient(
    base: CoefficientLike,
    *,
    tlist: ArrayLike = None,
    args: dict = {},
    args_ctypes: dict = {},
    order: int = 3,
    compile_opt: dict = None,
    function_style: str = None,
    boundary_conditions: tuple | str = None,
    **kwargs
):
    """Build ``Coefficient`` for time dependent systems:

    ```
    QobjEvo = Qobj + Qobj * Coefficient + Qobj * Coefficient + ...
    ```

    The coefficients can be a function, a string or a numpy array. Other
    packages may add support for other kind of coefficients.

    For function based coefficients, the function signature must be either:

    * ``f(t, ...)`` where the other arguments are supplied as ordinary
      "pythonic" arguments (e.g. ``f(t, w, a=5)``)
    * ``f(t, args)`` where the arguments are supplied in a "dict" named
      ``args``

    By default the signature style is controlled by the
    ``qutip.settings.core["function_coefficient_style"]`` setting, but it
    may be overriden here by specifying either ``function_style="pythonic"``
    or ``function_style="dict"``.

    *Examples*:

        - pythonic style function signature::

            def f1_t(t, w):
                return np.exp(-1j * t * w)

            coeff1 = coefficient(f1_t, args={"w": 1.})

        - dict style function signature::

            def f2_t(t, args):
                return np.exp(-1j * t * args["w"])

            coeff2 = coefficient(f2_t, args={"w": 1.})

    For string based coeffients, the string must be a compilable python code
    resulting in a complex. The following symbols are defined:

        sin, cos, tan, asin, acos, atan, pi,
        sinh, cosh, tanh, asinh, acosh, atanh,
        exp, log, log10, erf, zerf, sqrt,
        real, imag, conj, abs, norm, arg, proj,
        numpy as np,
        scipy.special as spe (python interface)
        and cython_special (scipy cython interface)

    *Examples*::

        coeff = coefficient('exp(-1j*w1*t)', args={"w1":1.})

    'args' is needed for string coefficient at compilation.
    It is a dict of (name:object). The keys must be a valid variables string.

    Compilation options can be passed as "compile_opt=CompilationOptions(...)".

    For numpy array format, the array must be an 1d of dtype float or complex.
    A list of times (float64) at which the coeffients must be given (tlist).
    The coeffients array must have the same len as the tlist.
    The time of the tlist do not need to be equidistant, but must be sorted.
    By default, a cubic spline interpolation will be used to compute the
    coefficient at time t. The keyword ``order`` sets the order of the
    interpolation. When ``order = 0``, the interpolation is step function that
    evaluates to the most recent value.

    *Examples*::

        tlist = np.logspace(-5,0,100)
        H = QobjEvo(np.exp(-1j*tlist), tlist=tlist)

    ``scipy.interpolate``'s ``CubicSpline``, ``PPoly`` and ``Bspline`` are
    also converted to interpolated coefficients (the same kind of coefficient
    created from ``ndarray``). Other interpolation methods from
    scipy are converted to a function-based coefficient (the same kind of
    coefficient created from callables).

    Parameters
    ----------
    base : object
        Base object to make into a Coefficient.

    args : dict, optional
        Dictionary of arguments to pass to the function or string coefficient.

    order : int, default=3
        Order of the spline for array based coefficient.

    tlist : iterable, optional
        Times for each element of an array based coefficient.

    function_style : str {"dict", "pythonic", None}, optional
        Function signature of function based coefficients.

    args_ctypes : dict, optional
        C type for the args when compiling array based coefficients.

    compile_opt : CompilationOptions, optional
        Sets of options for the compilation of string based coefficients.

    boundary_conditions: 2-tupule, str or None, optional
        Specify boundary conditions for spline interpolation.

    **kwargs
        Extra arguments to pass the the coefficients.
    """
    kwargs.update({
        "tlist": tlist,
        'args': args,
        'args_ctypes': args_ctypes,
        'order': order,
        'compile_opt': compile_opt,
        'function_style': function_style,
        'boundary_conditions': boundary_conditions
    })

    for type_ in coefficient_builders:
        if isinstance(base, type_):
            return coefficient_builders[type_](base, **kwargs)

    if callable(base):
        op = FunctionCoefficient(base, args.copy(), style=function_style)
        if not isinstance(op(0), numbers.Number):
            raise TypeError("The coefficient function must return a number")
        return op
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


def const(value):
    """ return a Coefficient with a constant value.
    """
    return ConstantCoefficient(value)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%      Everything under this is for string compilation      %%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
WARN_MISSING_MODULE = [0]


class CompilationOptions(QutipOptions):
    """
    Options that control compilation of string based coefficient to Cython.

    These options can be set globaly:

        ``settings.compile["compiler_flags"] = "-O1"``

    In a ``with`` block:

        ``with CompilationOptions(use_cython=False):``

    Or as an instance:

        ``coefficient(coeff, compile_opt=CompilationOptions(recompile=True))``

    ********************
    Compilation options:
    ********************

    use_cython: bool
        Whether to compile strings as cython code or use python's ``exec``.

    recompile : bool
        Do not use previously made files but build a new one.

    try_parse: bool [True]
        Whether to try parsing the string for reuse and static typing.

    static_types : bool [True]
        Whether to use C types for constant and args.

    accept_int : None, bool
        Whether to use the type ``int`` for integer constants and args or
        upgrade it to ``float`` or ``complex``.
        If `None`, it will only use ``int`` when subscription is found in the
        code.

    accept_float : bool
        Whether to use the type ``float`` or upgrade them to ``complex``.

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
        import or cimport code, ex:
        extra_import="from scipy.linalg import det"
        extra_import="from qutip.core.data cimport CSR"

    clean_on_error : bool [True]
        When writing a cython file that cannot be imported, erase it.

    build_dir: str [None]
        cythonize's build_dir.
    """
    _link_flags = ""
    _compiler_flags = ""
    if sys.platform == 'win32':
        _compiler_flags = ''
    elif sys.platform == 'darwin':
        _compiler_flags = '-w -O3 -funroll-loops -mmacosx-version-min=10.9'
        _link_flags += '-mmacosx-version-min=10.9'
    else:
        _compiler_flags = '-w -O3 -funroll-loops'

    try:
        import cython
        import filelock
        import setuptools
        _use_cython = True
    except ImportError:
        _use_cython = False
        WARN_MISSING_MODULE[0] = 1

    _options = {
        "use_cython": _use_cython,
        "try_parse": True,
        "static_types": True,
        "accept_int": None,
        "accept_float": None,
        "recompile": False,
        "compiler_flags": _compiler_flags,
        "link_flags": _link_flags,
        "extra_import": "",
        "clean_on_error": True,
        "build_dir": None,
    }
    _settings_name = "compile"


# Create the default instance in settings.
qset.compile = CompilationOptions()


# Version number of the Coefficient
COEFF_VERSION = "1.2"

try:
    root = os.path.join(qset.tmproot, f"qutip_coeffs_{COEFF_VERSION}")
    qset.coeffroot = root
except OSError:
    qset.coeffroot = "."


def clean_compiled_coefficient(all=False):
    """
    Remove previouly compiled string Coefficient.

    Parameter:
    ----------
    all: bool
        If not `all`, it will remove only previous version.
    """
    import glob
    import shutil
    tmproot = qset.tmproot
    active = qset.coeffroot
    folders = glob.glob(os.path.join(tmproot, 'qutip_coeffs_') + "*")
    if all:
        shutil.rmtree(active)
    for folder in folders:
        if folder != active:
            shutil.rmtree(folder)
    # Recreate the empty folder.
    qset.coeffroot = qset.coeffroot


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


def coeff_from_str(base, args, args_ctypes, compile_opt=None, **_):
    """
    Entry point for string based coefficients
    - Test if the string is valid
    - Parse: "cos(a*t)" and "cos( w1 * t )"
        should be recognised as the same compiled object.
    - Verify if already compiled and compile if not
    """
    # First, a sanity check before thinking of compiling
    if compile_opt is None:
        compile_opt = qset.compile
    if not compile_opt['extra_import']:
        try:
            env = {"t": 0}
            env.update(args)
            exec(base, str_env, env)
        except Exception as err:
            raise Exception("Invalid string coefficient") from err
    coeff = None
    # Do we compile?
    if not compile_opt['use_cython']:
        if WARN_MISSING_MODULE[0]:
            warnings.warn(
                "`cython`, `setuptools` and `filelock` are required for "
                "compilation of string coefficents. Falling back on `eval`.")
            # Only warns once.
            WARN_MISSING_MODULE[0] = 0
        return StrFunctionCoefficient(base, args)
    # Parsing tries to make the code in common pattern
    parsed, variables, constants, raw = try_parse(base, args,
                                                  args_ctypes, compile_opt)
    # Once parsed, the code should be unique enough to get a filename
    hash_ = hashlib.sha256(bytes(parsed, encoding='utf8'))
    file_name = "qtcoeff_" + hash_.hexdigest()[:30]
    # See if it already exist and import it.
    if not compile_opt['recompile']:
        coeff = try_import(file_name, parsed)

    if not coeff and qset.coeff_write_ok:
        # Previously compiled coefficient not available: create the cython code
        code = make_cy_code(parsed, variables, constants,
                            raw, compile_opt)
        try:
            coeff = compile_code(code, file_name, parsed, compile_opt)
        except PermissionError:
            pass
    if coeff is None:
        # We don't use cython or compilation failed
        return StrFunctionCoefficient(base, args)
    keys = [key for _, key, _ in variables]
    const = [fromstr(val) for _, val, _ in constants]
    return coeff(base, keys, const, args)


coefficient_builders[str] = coeff_from_str


def try_import(file_name, parsed_in):
    """ Import the compiled coefficient if existing and check for
    name collision.
    """
    try:
        mod = importlib.import_module(file_name)
    except ModuleNotFoundError:
        # Coefficient does not exist, to compile as file_name
        return None

    if mod.parsed_code == parsed_in:
        # Coefficient found!
        return mod.StrCoefficient
    else:
        raise ValueError("string hash collision, change the string "
                         "or clean files in qutip.settings.coeffroot")


def make_cy_code(code, variables, constants, raw, compile_opt):
    """
    Generate the code for the string coefficients.
    """
    cdef_cte = ""
    init_cte = ""
    copy_cte = ""
    for i, (name, val, ctype) in enumerate(constants):
        cdef_cte += "        {} {}\n".format(ctype, name[5:])
        copy_cte += "        out.{} = {}\n".format(name[5:], name)
        init_cte += "        {} = cte[{}]\n".format(name, i)
    cdef_var = ""
    init_var = ""
    init_arg = ""
    replace_var = ""
    call_var = ""
    copy_var = ""
    for i, (name, val, ctype) in enumerate(variables):
        cdef_var += "        str key{}\n".format(i)
        cdef_var += "        {} {}\n".format(ctype, name[5:])
        copy_var += "        out.key{} = self.key{}\n".format(i, i)
        copy_var += "        out.{} = {}\n".format(name[5:], name)
        if not raw:
            init_var += "        self.key{} = var[{}]\n".format(i, i)
        else:
            init_var += "        self.key{} = '{}'\n".format(i, val)
        init_arg += "        {} = args[self.key{}]\n".format(name, i)
        replace_var += "            if self.key{} in kwargs:\n".format(i)
        replace_var += ("                out.{}"
                        " = kwargs[self.key{}]\n".format(name[5:], i))
        if raw:
            call_var += "        cdef {} {} = {}\n".format(ctype, val, name)

    code = f"""#cython: language_level=3
# This file is generated automatically by QuTiP.

import numpy as np
import scipy.special as spe
from scipy.special cimport cython_special
cimport cython
from qutip.core.cy.coefficient cimport Coefficient
from qutip.core.cy.math cimport erf, zerf
from qutip.core.cy.complex_math cimport *
from qutip.core.data cimport Data
cdef double pi = 3.14159265358979323
{compile_opt['extra_import']}

parsed_code = "{code}"

@cython.auto_pickle(True)
cdef class StrCoefficient(Coefficient):
    \"\"\"
    String compiled as a :obj:`.Coefficient` using cython.
    \"\"\"
    cdef:
        str codeString
{cdef_cte}{cdef_var}

    def __init__(self, base, var, cte, args):
        self.codeString = base
{init_cte}{init_var}{init_arg}

    cpdef Coefficient copy(self):
        \"\"\"Return a copy of the :obj:`.Coefficient`.\"\"\"
        cdef StrCoefficient out = StrCoefficient.__new__(StrCoefficient)
        out.codeString = self.codeString
{copy_cte}{copy_var}
        return out

    def replace_arguments(self, _args=None, **kwargs):
        \"\"\"
        Return a :obj:`.Coefficient` with args changed for :obj:`.Coefficient`
        built from 'str' or a python function. Or a the :obj:`.Coefficient`
        itself if the :obj:`.Coefficient` does not use arguments. New arguments
        can be passed as a dict or as keywords.

        Parameters
        ----------
        _args : dict
            Dictionary of arguments to replace.

        **kwargs
            Arguments to replace.
        \"\"\"
        cdef StrCoefficient out

        if _args:
            kwargs.update(_args)
        if kwargs:
            out = self.copy()
{replace_var}
            return out
        return self

    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cdef complex _call(self, double t) except *:
{call_var}        return {code}
"""
    return code


def compile_code(code, file_name, parsed, c_opt):
    pwd = os.getcwd()
    os.chdir(qset.coeffroot)
    # Files with the same name, but differents extension than the pyx file, are
    # erased during cythonization process, breaking filelock.
    # Adding a prefix make them safe to use.
    lock = filelock.FileLock("compile_lock_" + file_name + ".lock")
    try:
        lock.acquire(timeout=0)
        for file in glob.glob(file_name + "*"):
            os.remove(file)
        file_ = open(file_name + ".pyx", "w")
        file_.writelines(code)
        file_.close()
        oldargs = sys.argv
        try:
            sys.argv = ["setup.py", "build_ext", "--inplace"]
            coeff_file = Extension(
                file_name,
                sources=[file_name + ".pyx"],
                extra_compile_args=c_opt['compiler_flags'].split(),
                extra_link_args=c_opt['link_flags'].split(),
                include_dirs=[np.get_include()],
                language='c++'
            )
            ext_modules = cythonize(
                coeff_file, force=True, build_dir=c_opt['build_dir']
            )
            setup(ext_modules=ext_modules)
        except Exception as e:
            if c_opt['clean_on_error']:
                for file in glob.glob(file_name + "*"):
                    os.remove(file)
            raise Exception("Could not compile") from e
        finally:
            sys.argv = oldargs
    except filelock.Timeout:
        with lock:
            # We wait for the lock to be released and then retry the import.
            pass
    finally:
        lock.release()
        os.chdir(pwd)
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
    accept_int = compile_opt['accept_int']
    accept_float = compile_opt['accept_float']
    if accept_int is None:
        # If there is a subscript: a[b] int are always accepted to be safe
        # with TypeError.
        # Also comparison is not supported for complex.
        accept_int = "SUBSCR" in dis.Bytecode(code).dis()
    if accept_float is None:
        accept_float = "COMPARE_OP" in dis.Bytecode(code).dis()
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


def use_hinted_type(variables, code, args_ctypes):
    variables_manually_typed = []
    for i, (name, key, type_) in enumerate(variables):
        if key in args_ctypes:
            new_name = "self._custom_" + args_ctypes[key] + str(i)
            code = code.replace(name, new_name)
            variables_manually_typed.append((new_name, key, args_ctypes[key]))
        else:
            variables_manually_typed.append((name, key, type_))
    return code, variables_manually_typed


def try_parse(code, args, args_ctypes, compile_opt):
    """
    Try to parse and verify that the result is still usable.
    """
    if not compile_opt['try_parse']:
        variables = [("self." + name, name, "object") for name in args
                     if name in code]
        code, variables = use_hinted_type(variables, code, args_ctypes)
        return code, variables, [], True
    ncode, variables, constants = parse(code, args, compile_opt)
    if not compile_opt['static_types']:
        # Fallback to object
        variables = [(f, s, "object") for f, s, _ in variables]
        constants = [(f, s, "object") for f, s, _ in constants]
    ncode, variables = use_hinted_type(variables, ncode, args_ctypes)
    if (
        (compile_opt['extra_import']
         and not compile_opt['extra_import'].isspace())
        or test_parsed(ncode, variables, constants, args)
    ):
        return ncode, variables, constants, False
    else:
        warnings.warn("Could not find c types", StringParsingWarning)
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
    except Exception:
        return False
    return True
