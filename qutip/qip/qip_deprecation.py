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
