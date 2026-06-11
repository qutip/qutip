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
    feedback : callable, ``f(keys : set) -> None``, optional
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
        if feedback is None:
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
        if key not in self._default:
            raise KeyError(f"Options {key} is not supported.")
        super().__setitem__(key, self._default[key])
        if (updater := self._on_update()) is not None:
            updater(key)

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
    def _from_reduced(cls, default, feedback, name, doc, keys, args):
        return cls(default, feedback, name, doc, **{
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
