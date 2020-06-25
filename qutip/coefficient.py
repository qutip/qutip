import numpy as np
import scipy
import os
import sys
import re
import dis
import hashlib
import glob
import importlib
import qutip.settings as qset
from qutip import Cubic_Spline
from qutip.cy.coefficient import (InterpolateCoefficient, InterCoefficient,
                                  StepCoefficient, FunctionCoefficient)
from setuptools import setup, Extension
from Cython.Build import cythonize


def coefficient(base, tlist=None, args={}, _stepInterpolation=False):
    if isinstance(base, Cubic_Spline):
        return InterpolateCoefficient(base)

    elif isinstance(base, np.ndarray):
        if len(base.shape) != 1:
            raise ValueError("The array to interpolate must be a 1D array")
        if base.shape != tlist.shape:
            raise ValueError("tlist must be the same len as the array to interpolate")
        base = base.astype(np.complex128)
        tlist = tlist.astype(np.float64)
        if not _stepInterpolation:
            return InterCoefficient(base, tlist)
        else:
            return StepCoefficient(base, tlist)

    elif isinstance(base, str):
        return coeff_from_str(base, args)
    elif callable(base):
        # TODO add tests?
        return FunctionCoefficient(base)


# Everything under if for string compilation
_link_flags = []
if (sys.platform == 'win32'
    and int(str(sys.version_info[0])+str(sys.version_info[1])) >= 35
    and os.environ.get('MSYSTEM') is None):
    _compiler_flags = ['/w', '/Ox']
# Everything else
else:
    _compiler_flags = ['-w', '-O3', '-funroll-loops']
    if sys.platform == 'darwin':
        # These are needed for compiling on OSX 10.14+
        _compiler_flags.append('-mmacosx-version-min=10.9')
        _link_flags.append('-mmacosx-version-min=10.9')


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


typeCodes = [
    ("double[::1]", "_adbl"),
    ("complex[::1]", "_acpl"),
    ("complex", "_cpl"),
    ("double", "_dbl"),
    ("int", "_int"),
    ("str", "_str"),
    ("object", "_obj")
]


def coeff_from_str(base, args, use_cython=True, _debug=False):
    # First a sanity check before thinking of compiling
    try:
        env = {"t": 0}
        env.update(args)
        exec(base, str_env, env)
    except Exception as err:
        raise Exception("Invalid string coefficient") from err
    # Do we even compile?
    if not qset.use_cython or not use_cython:
        return _str_as_func(base, args)
    # Parsing tries to make the code in common pattern
    parsed, variables, constants = try_parse(base, args)
    if _debug:
        print(parsed, variables, constants)
    # Once parsed, the code should be unique enough to get a filename
    hash_ = hashlib.sha256(bytes(parsed, encoding='utf8'))
    file_name = "qtcoeff_" + hash_.hexdigest()[:30]
    if _debug:
        print(file_name)
    # See if it already exist, if not write and cythonize it
    coeff = try_import(file_name, parsed)
    if coeff is None:
        code = make_cy_code(parsed, variables, constants, base)
        if _debug:
            print(code)
        coeff = compile_code(code, file_name, parsed)
    keys = [key for _, key, _ in variables]
    const = [fromstr(val) for _, val, _ in constants]
    return coeff(keys, const)


def try_import(file_name, parsed_in):
    coeff = None
    try:
        mod = importlib.import_module(file_name)
        coeff = getattr(mod, "StrCoefficient")
        parsed_saved = getattr(mod, "parsed_code")
    except Exception as e:
        print(e)
        parsed_saved = ""
    if parsed_in != parsed_saved:
        # hash collision!
        coeff = None
        print("hash!?")
    return coeff


def make_cy_code(code, variables, constants, base):
    #_cython_path = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
    _cython_path = os.path.dirname(os.path.abspath(__file__)).replace(
                    "\\", "/")
    _include_string = "'" + _cython_path + "/cy/complex_math.pxi'"

    cdef_cte = ""
    init_cte = ""
    get_cte = ""
    set_cte = ""
    for i, (name, val, ctype) in enumerate(constants):
        cdef_cte += "        {} {}\n".format(ctype, name[5:])
        init_cte += "        {} = cte[{}]\n".format(name, i)
        get_cte += "             {},\n".format(name)
        set_cte += "        {} = state[{}]\n".format(name, i)
    cdef_var = ""
    init_var = ""
    call_var = ""
    get_var = ""
    set_var = ""
    for i, (name, val, ctype) in enumerate(variables):
        cdef_var += "        str key{}\n".format(i)
        init_var += "        self.key{} = var[{}]\n".format(i, i)
        call_var += "        cdef {} {} = args[self.key{}]\n".format(ctype, name, i)
        get_var += "             self.key{},\n".format(i)
        set_var += "        self.key{} = state[{}]\n".format(i, i + len(constants))

    code = """#cython: language_level=3
# This file is generated automatically by QuTiP.

import numpy as np
import scipy.special as spe
cimport cython
from qutip.cy.coefficient cimport Coefficient
from qutip.cy.math cimport erf, zerf
cdef double pi = 3.14159265358979323
include {}


parsed_code = "{}"


cdef class StrCoefficient(Coefficient):
    cdef:
        int dummy
{}{}

    def __init__(self, var, cte):
        self.codeString = "{}"
{}{}
    cdef complex _call(self, double t, dict args):
{}
        return {}

    def __getstate__(self):
        return (
{}{}               )

    def __setstate__(self, state):
{}{}
        pass
""".format(_include_string, code, cdef_cte, cdef_var, base,
           init_cte, init_var, call_var, code,
           get_cte, get_var, set_cte, set_var
          )
    return code


def compile_code(code, file_name, parsed):
    root = qset.tmproot
    full_file_name = os.path.join(root, file_name)
    file_ = open(full_file_name + ".pyx", "w")
    file_.writelines(code)
    file_.close()
    oldargs = sys.argv
    try :
        sys.argv = ["setup.py", "build_ext", "--inplace"]
        coeff_file = Extension(file_name,
                               sources=[full_file_name + ".pyx"],
                               extra_compile_args=_compiler_flags,
                               extra_link_args=_link_flags,
                               language='c++')
        setup(ext_modules = cythonize(coeff_file))
        libfile = glob.glob(file_name + "*")[0]
        os.rename(libfile, os.path.join(root, libfile))
    except Exception as e:
        raise Exception("Could not compile") from e
    sys.argv = oldargs
    return try_import(file_name, parsed)


def compileType(value):
    if isinstance(value, np.ndarray) and isinstance(value[0], (float, np.float32, np.float64)):
        ctype = 0
    elif isinstance(value, np.ndarray) and isinstance(value[0], (complex, np.complex128)):
        ctype = 1
    elif isinstance(value, (complex, np.complex128)):
        ctype = 2
    elif isinstance(value, (float, np.float32, np.float64)):
        ctype = 3
    elif np.isscalar(value):
        ctype = 4
    elif isinstance(value, (str)):
        ctype = 5
    else:
        ctype = 6
    return ctype


def find_type_from_str(chars):
    try:
        lc = {}
        exec("out = " + chars, {}, lc)
        return compileType(lc["out"])
    except Exception:
        return None


def fix_type(ctype, accept_int, accept_float):
    if ctype == 4 and not accept_int:
        ctype = 3
    if ctype == 3 and not accept_float:
        ctype = 2
    return ctype


def extract_constant(code):
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
    const_strs = re.findall(pattern, code)
    for cte in const_strs:
        name = " _cteX{}_ ".format(len(constants))
        code = code.replace(cte, cte[0] + name, 1)
        constants.append(( name[1:-1], cte[1:], find_type_from_str(cte[1:]) ))
    return code


def space_parts(code, names):
    for name in names:
        code = re.sub("(?<=[^0-9a-zA-Z_])" + name + "(?=[^0-9a-zA-Z_])",
                      " " + name + " ", code)
    code = " ".join(code.split())
    return code


def parse(code, args, accept_int=None, accept_float=True):
    """
    Read the code and change variables names and constant so
    '2.*cos(w1*t)' and '1.5 * cos( w2 * t )' be compiled as the same
    Coefficient object.
    """
    code, constants = extract_constant(code)
    names = re.findall("[0-9a-zA-Z_]+", code)
    code = space_parts(code, names)
    constants_names = [const[0] for const in constants]
    new_code = []
    ordered_constants = []
    variables = []
    typeCounts = [0,0,0,0,0,0,0]
    if accept_int is None:
        accept_int = "SUBSCR" in dis.Bytecode(code).dis()
    for word in code.split():
        if word not in names:
            # syntax
            new_code.append(word)
        elif word in args:
            ctype = compileType(args[word])
            ctype = fix_type(ctype, accept_int, accept_float)
            var_name = typeCodes[ctype][1] + str(typeCounts[ctype])
            typeCounts[ctype] += 1
            new_code.append(var_name)
            variables.append((var_name, word, typeCodes[ctype][0]))
        elif word in constants_names:
            name, val, ctype = constants[int(word[5:-1])]
            ctype = fix_type(ctype, accept_int, accept_float)
            cte_name = "self._cte" + typeCodes[ctype][1] +\
                       str(len(ordered_constants))
            new_code.append(cte_name)
            ordered_constants.append((cte_name, val, typeCodes[ctype][0]))
        else:
            # Hopefully a buildin or known object
            new_code.append(word)
        code = " ".join(new_code)
    return code, variables, ordered_constants


def try_parse(code, args, accept_int=None, accept_float=True):
    ncode, variables, ordered_constants = parse(code, args,
                                                accept_int, accept_float)
    if test_parsed(ncode, variables, ordered_constants, args):
        return ncode, variables, ordered_constants
    else:
        variables = [(name, name, ctype) for _, name, ctype in variables]
        return code, variables, []


def test_parsed(code, variables, constants, args):
    cte_dict = {cte[0]:fromstr(cte[1]) for cte in constants}
    var_dict = {var[0]:args[var[1]] for var in variables}
    var_dict.update(cte_dict)
    var_dict["t"] = 1
    try:
        exec(code, str_env, var_dict)
    except Exception:
        return False
    return True


def fromstr(base):
    ls = {}
    exec("out = "+ base, {}, ls)
    return ls["out"]
