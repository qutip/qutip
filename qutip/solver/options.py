__all__ = []

import warnings
import weakref


class _SolverOptions(dict):
    """
    Class to hold options for solver and integrator.

    Parameters
    ----------
    default : dict
        Default dict, only keys in this will be accepted.
    on_update : callable, ``f(keys : set) -> None``, optional
        Function to called when an item is updated.
    name : str, optional
        Name of the solver or integrator that use this. Used in __repr__ only.
    doc : str, optional
        Overwrite the __doc__ of the instance.
    """
    def __init__(
        self, default, on_update=None, name="", doc="", /, **kwargs
    ):
        self._default = default
        self.__doc__ = doc
        if on_update is None:
            self._on_update = lambda : None
        else:
            self._on_update = weakref.WeakMethod(on_update)
        self._name = name
        extra_keys = kwargs.keys() - default.keys()
        if extra_keys:
            raise KeyError(f"Options {extra_keys} are not supported.")
        super().__init__(**{**self._default, **kwargs})

    def __setitem__(self, key, val):
        if key not in self._default:
            raise KeyError(f"Options {key} is not supported.")
        if val is None:
            val = self._default[key]
        if val == self[key]:
            return
        super().__setitem__(key, val)
        if (updater := self._on_update()) is not None:
            updater(key)

    def __delitem__(self, key):
        self[key] = None

    def pop(self):
        raise RuntimeError("Can't remove options")

    def popitem(self):
        raise RuntimeError("Can't remove options")

    def clear(self):
        raise RuntimeError("Can't remove options")

    def update(self, *args, **kwargs):
        tmp = {}
        tmp.update(*args, **kwargs)
        for key, val in tmp.items():
            # Ensure all keys are present
            self[key] = val

    def copy(self):
        return self.__class__(
            self._default,
            None,
            self._name,
            self.__doc__,
            **self
        )

    def __str__(self):
        lines = []
        keys = [key for key in self.keys()]
        vals = [repr(val) for val in self.values()]
        pad_key = max(len(key) for key in keys)
        pad_val = max(len(val) for val in vals) + 3
        lines.append(f"Options for {self._name}:")
        for key, val in zip(keys, vals):
            default = "(default)" if self[key] == self._default[key] else ""
            lines.append(f"    {key:{pad_key}} : {val:<{pad_val}}{default}")
        return "\n".join(lines)

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text(f"_SolverOptions(...)")
        else:
            p.text(self.__str__())

    @classmethod
    def _from_reduced(cls, default, on_update, name, doc, keys, args):
        return cls(default, on_update, name, doc, **{
            key: arg for key, arg in zip(keys, args)
        })

    def __reduce__(self):
        return (
            self._from_reduced,
            (
                self._default,
                None,
                self._name,
                self.__doc__,
                tuple(self.keys()),
                tuple(self.values())
                )
            )



###############################################################################
import weakref
from typing import TypedDict, ClassVar, Optional
from dataclasses import dataclass, MISSING, KW_ONLY, asdict, field

from qutip import SESolver


class _SolverDataOptions:
    _on_update = staticmethod(lambda: None)

    def _connect(self, solver):
        """Binds a weak reference callback to the parent solver."""
        self._on_update = weakref.WeakMethod(solver._apply_options)

    @property
    def _name(self):
        # Strips 'rOptions' from class name: SESolverOptions -> sesolve
        return self.__class__.__name__[:-8].lower()

    def __setattr__(self, key, val):
        _need_update = False
        defaults = getattr(self, '_default', {})
        if key in defaults:
            if val is None:
                val = defaults[key]
            if val != getattr(self, key, None):
                _need_update = True
        super().__setattr__(key, val)
        if _need_update and (updater := self._on_update()) is not None:
            updater(key)

    def __delattr__(self, key):
        if key in self._default:
            self.__setattr__(key, None)
        else:
            super().__delattr__(key)

    def __str__(self):
        lines = []
        keys = [key for key in self._default.keys()]
        vals = [repr(getattr(self, key)) for key in keys]
        pad_key = max(len(key) for key in keys)
        pad_val = max(len(val) for val in vals) + 3
        lines.append(f"Options for {self._name}:")
        for key, val in zip(keys, vals):
            default = "(default)" if getattr(self, key) == self._default[key] else ""
            lines.append(f"    {key:{pad_key}} : {val:<{pad_val}}{default}")
        return "\n".join(lines)

    def _repr_pretty_(self, p, cycle):
        if cycle:
            p.text(f"{self.__class__.__name__}(...)")
        else:
            p.text(self.__str__())


class PBarSubOptions(TypedDict):
    chunk_size: int


@dataclass(repr=False)
class SESolverOptions(_SolverDataOptions):
    _ : KW_ONLY
    progress_bar: Optional[str] = None
    progress_kwargs: Optional[PBarSubOptions] = None
    store_final_state: Optional[bool] = None
    store_states: Optional[bool] = None
    normalize_output: Optional[bool] = None
    method: Optional[str] = None

    _default = SESolver.solver_options
    __doc__ = SESolver.options.__doc__
