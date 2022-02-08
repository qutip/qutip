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
# Running on mac with openblas make eigh unsafe
eigh_unsafe = False
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
    val = __self[key]
    if isinstance(val, (bool, int, float, complex, str)):
        return True
    return False


_environment_keys = ["ipython", 'has_mkl', 'has_openmp',
                     'mkl_lib', 'fortran', 'num_cpus']
__self = locals().copy()  # Not ideal, making an object would be better
__all_out = [key for key in __self if _valid_config(key)]
__all = [key for key in __all_out if key not in _environment_keys]
__default = {key: __self[key] for key in __all}
__section = "qutip"
del _valid_config
__self = locals()


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
    for key in __default:
        __self[key] = __default[key]

    from qutip.utilities import available_cpu_count
    __self["num_cpus"] = available_cpu_count()

    try:
        from qutip.cy.openmp.parfuncs import spmv_csr_openmp
    except:
        __self["has_openmp"] = False
        __self["openmp_thresh"] = 10000
    else:
        __self["has_openmp"] = True
        from qutip.cy.openmp.bench_openmp import calculate_openmp_thresh
        thrsh = calculate_openmp_thresh()
        __self["openmp_thresh"] = thrsh

    try:
        __IPYTHON__
        __self["ipython"] = True
    except:
        __self["ipython"] = False

    from qutip._mkl.utilities import _set_mkl
    _set_mkl()


def __repr__():
    out = "qutip settings:\n"
    longest = max(len(key) for key in __all_out)
    for key in __all_out:
        out += "{:{width}} : {}\n".format(key, __self[key], width=longest)
    return out
