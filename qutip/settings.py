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
"""
This module contains settings for the QuTiP graphics, multiprocessing, and
tidyup functionality, etc.
"""
from __future__ import absolute_import
# use auto tidyup
auto_tidyup = True
# use auto tidyup dims on multiplication
auto_tidyup_dims = True
# detect hermiticity
auto_herm = True
# general absolute tolerance
atol = 1e-12
# use auto tidyup absolute tolerance
auto_tidyup_atol = 1e-12
# number of cpus (set at qutip import)
num_cpus = 0
# flag indicating if fortran module is installed
# never used
fortran = False
# path to the MKL library
mkl_lib = None
# Flag if mkl_lib is found
has_mkl = False
# Has OPENMP
has_openmp = False
# debug mode for development
debug = False
# are we in IPython? Note that this cannot be
# set by the RC file.
ipython = False
# define whether log handler should be
#   - default: switch based on IPython detection
#   - stream: set up non-propagating StreamHandler
#   - basic: call basicConfig
#   - null: leave logging to the user
log_handler = 'default'
# Allow for a colorblind mode that uses different colormaps
# and plotting options by default.
colorblind_safe = False
# Sets the threshold for matrix NNZ where OPENMP
# turns on. This is automatically calculated and
# put in the qutiprc file.  This value is here in case
# that failts
openmp_thresh = 10000
# Note that since logging depends on settings,
# if we want to do any logging here, it must be manually
# configured, rather than through _logging.get_logger().
try:
    import logging
    _logger = logging.getLogger(__name__)
    _logger.addHandler(logging.NullHandler())
    del logging  # Don't leak names!
except:
    _logger = None


def _valid_config(key):
    if key == "absolute_import":
        return False
    if key.startswith("_"):
        return False
    val = _self[key]
    if isinstance(val, (bool, int, float, complex, str)):
        return True
    return False


_environment_keys = ["ipython", 'has_mkl', 'has_openmp',
                     'mkl_lib', 'fortran', 'num_cpus']
_self = locals().copy()  # Not ideal, making an object would be better
_all_out = [key for key in _self if _valid_config(key)]
_all = [key for key in _all_out if key not in _environment_keys]
_default = {key: _self[key] for key in _all}
del _valid_config
_self = locals()


def save(file='qutiprc', all_config=True):
    """
    Write the settings to a file.
    Default file is 'qutiprc' which is loaded when importing qutip.
    File are stored in .qutip directory in the user home.
    The file can be a full path or relative to home to save elsewhere.
    If 'all_config' is used, also load other available configs.
    """
    from qutip.configrc import write_rc_qset, write_rc_config
    if all_config:
        write_rc_config(file)
    else:
        write_rc_qset(file)


def load(file='qutiprc', all_config=True):
    """
    Loads the settings from a file.
    Default file is 'qutiprc' which is loaded when importing qutip.
    File are stored in .qutip directory in the user home.
    The file can be a full path or relative to home to save elsewhere.
    If 'all_config' is used, also load other available configs.
    """
    from qutip.configrc import load_rc_qset, load_rc_config
    if all_config:
        load_rc_config(file)
    else:
        load_rc_qset(file)


def reset():
    """Hard reset of the qutip.settings values
    Recompute the threshold for openmp, so it may be slow.
    """
    for key in _default:
        _self[key] = _default[key]

    import os
    if 'QUTIP_NUM_PROCESSES' in os.environ:
        num_cpus = int(os.environ['QUTIP_NUM_PROCESSES'])
    elif 'cpus' in info:
        import qutip.hardware_info
        info = qutip.hardware_info.hardware_info()
        num_cpus = info['cpus']
    else:
        try:
            num_cpus = multiprocessing.cpu_count()
        except:
            num_cpus = 1
    _self["num_cpus"] = num_cpus

    try:
        from qutip.cy.openmp.parfuncs import spmv_csr_openmp
    except:
        _self["has_openmp"] = False
        _self["openmp_thresh"] = 10000
    else:
        _self["has_openmp"] = True
        from qutip.cy.openmp.bench_openmp import calculate_openmp_thresh
        thrsh = calculate_openmp_thresh()
        _self["openmp_thresh"] = thrsh

    try:
        __IPYTHON__
        _self["ipython"] = True
    except:
        _self["ipython"] = False

    from qutip._mkl.utilities import _set_mkl
    _set_mkl()


def list():
    out = "qutip settings:\n"
    longest = max(len(key) for key in _all_out)
    for key in _all_out:
        out += "{:{width}} : {}\n".format(key, _self[key], width=longest)
    return out


def QtOptionClass(name):
    """Make the class an Options object of Qutip and register the object
    default as qutip.settings."name".

    Add the methods:
        __init__:
            Allow to create from data in files or from default with attributes
            overwritten by keywords.
            Properties with setter can also be set as kwargs.
        save(file), load(file), reset():
            Save, load, reset will affect all attributes that can be saved
            as defined in qutip.configrc.getter.
        __repr__():
            Make a clean print of all attribute and properties.
    and the attributes:
        _all
        _repr_keys
        _name
        _fullname
        _isDefault
        _defaultInstance

    * Any attribute starting with "_" are excluded.

    Usage:
        ``
        @QtOptionClass(name)
        class Options:
            ...
        ``
    or
        ``
        @QtOptionClass
        class Options
            ...
        ``
    * default name is `Options.__name__`
    """
    # The real work is in _QtOptionMaker
    if isinstance(name, str):
        # Called as
        # @QtOptionClass(name)
        # class Options:
        return _QtOptionMaker(name)
    else:
        # Called as
        # @QtOptionClass
        # class Options:
        return _QtOptionMaker(name.__name__)(name)


class _QtOptionMaker:
    def __init__(self, name):
        self.name = name

    def __call__(self, cls):
        if hasattr(cls, "_isDefault"):
            # Already a QtOptionClass
            if self.name not in __self:
                self._make_default(cls)
            return

        # attributes that to be saved
        cls._all = [key for key in cls.__dict__
                    if self._valid(cls, key)]
        # attributes to print
        cls._repr_keys = [key for key in cls.__dict__
                          if self._valid(cls, key, _repr=True)]
        # Name in settings and in files
        cls._name = self.name
        # Name when printing
        cls._fullname = ".".join([cls.__module__, cls.__name__])
        # Is this instance the default for the other.
        cls._isDefault = False
        # Build the default instance
        # Do it before setting __init__ since it use this default
        self._make_default(cls)

        # add methods
        # __init__ is dynamically build to get a meaningful signature
        _all_set = [key for key in cls.__dict__
                    if self._valid(cls, key, _set=True)]
        attributes_kw = ["             {}=None,".format(var)
                         for var in _all_set]
        attributes_kw[-1] = attributes_kw[-1][:-1]
        attributes_set = ["    self.{0} = {0} if {0} is not None "
                          "else self._defaultInstance.{0}".format(var)
                          for var in _all_set]
        code = ["def __init__(self, file='', *,"] + \
            attributes_kw + \
            ["            ):"] + \
            attributes_set + \
            ["    if file:"] + \
            ["        self.load(file)"]
        code = "\n".join(code)
        ns = {}
        exec(code, globals(), ns)
        cls.__init__ = ns["__init__"]
        cls.__repr__ = _qoc_repr_
        cls.reset = _qoc_reset
        cls.save = _qoc_save
        cls.load = _qoc_load
        return cls

    @staticmethod
    def _valid(cls, key, _repr=False, _set=False):
        # Can it be saved, printed, initialed?
        import qutip.configrc as qrc
        if key.startswith("_"):
            return False
        data = getattr(cls, key)
        if _repr and isinstance(data, property):
            # Print all properties
            return True
        if _set and isinstance(data, property) and data.fset is not None:
            # Properties with a setter can be set in __init__
            return True
        # Only these types can be saved
        return type(data) in qrc.getter

    def _make_default(self, cls):
        import qutip.configrc as qrc
        default = cls()
        for key in cls._all:
            default.__dict__[key] = cls.__dict__[key]
        default._isDefault = True
        default._fullname = "qutip.settings." + self.name
        _self[self.name] = default
        cls._defaultInstance = default
        qrc.sections.append((self.name, default))


def _qoc_repr_(self):
    out = self._fullname + ":\n"
    longest = max(len(key) for key in self._repr_keys)
    for key in self._repr_keys:
        out += "{:{width}} : {}\n".format(key, getattr(self, key),
                                          width=longest)
    return out


def _qoc_reset(self):
    """Reset instance to the default value or the default to Qutip's default"""
    if self._isDefault:
        [setattr(self, key, getattr(self.__class__, key))
         for key in self._all]
    else:
        [setattr(self, key, getattr(self._defaultInstance, key))
         for key in self._all]


def _qoc_save(self, file="qutiprc"):
    """Save to desired file. 'qutiprc' if not specified"""
    import qutip.configrc as qrc
    qrc.write_rc_object(file, self._name, self)


def _qoc_load(self, file="qutiprc"):
    """Load from desired file. 'qutiprc' if not specified"""
    import qutip.configrc as qrc
    qrc.load_rc_object(file, self._name, self)
