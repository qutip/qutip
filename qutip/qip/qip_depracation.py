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
import warnings
import inspect
from functools import wraps as _func_wrap

from . import circuit
from . import qubits
from .operations import gates


__all__ = circuit.__all__ + qubits.__all__ + gates.__all__

# modules that are wrapped with deprecation warning
module_list = [gates, circuit, qubits]


def _qip_importation_warning():
    warnings.warn(
        "Importing functions/classes of the qip submodule directly from "
        "the namespace qutip is deprecated. "
        "Please import them from the submodule instead, e.g.\n"
        "from qutip.qip.operations import cnot\n"
        "from qutip.qip.circuit import QubitCircuit\n",
        DeprecationWarning, stacklevel=3)


def _qip_func_wrapper(func):
    """Function wrapper for adding a deprecation warning."""
    @_func_wrap(func)
    def deprecated_func(*args, **kwargs):
        _qip_importation_warning()
        return func(*args, **kwargs)
    return deprecated_func


for module in module_list:
# Wrap all qip functions with a deprecation warning
    _func_pairs = inspect.getmembers(module, inspect.isfunction)
    for _name, _func in _func_pairs:
        locals()[_name] = _qip_func_wrapper(_func)
del _name, _func, _func_pairs, _qip_func_wrapper


def _qip_class_wrapper(original_cls):
    """Class wrapper for adding a deprecation warning."""
    class Deprecated_cls(original_cls):
        def __init__(self, *args, **kwargs):
            _qip_importation_warning()
            super(Deprecated_cls, self).__init__(*args, **kwargs)
    # copy information form the original class, similar to functools.wraps
    for attr in ('__module__', '__name__', '__qualname__', '__doc__'):
        try:
            value = getattr(original_cls, attr)
        except AttributeError:
            pass
        else:
            setattr(Deprecated_cls, attr, value)
    return Deprecated_cls


# Wrap all qip classes with a deprecation warning
for module in module_list:
    _cls_pairs = inspect.getmembers(module, inspect.isclass)
    for _name, _cls in _cls_pairs:
        locals()[_name] = _qip_class_wrapper(_cls)
del _name, _cls, _cls_pairs, _qip_class_wrapper

del module_list
