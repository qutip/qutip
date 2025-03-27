__all__ = ["Options", "SolverOptions"]

import warnings


def SolverOptions(*args, **kwargs):
    warnings.warn(
        "Dedicated options class are no longer needed, "
        "options should be passed as dict to solvers.",
        FutureWarning
    )
    return kwargs


def Options(**kwargs):
    warnings.warn(
        "Dedicated options class are no longer needed, "
        "options should be passed as dict to solvers.",
        FutureWarning
    )
    return kwargs


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
        self, default, feedback=None, name="", doc="", /, **kwargs
    ):
        self._default = default
        self.__doc__ = doc
        self._feedback = feedback
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
        if self._feedback:
            self._feedback(key)

    def __delitem__(self, key):
        if key not in self._default:
            raise KeyError(f"Options {key} is not supported.")
        super().__setitem__(key, self._default[key])
        if self._feedback:
            self._feedback(key)

    def copy(self):
        return self.__class__(
            self._default,
            self._feedback,
            self._name,
            self.__doc__,
            **self
        )

    def __str__(self):
        lines = []
        longest = max(len(key) for key in self.keys())
        lines.append(f"Options for {self._name}:")
        for key in self.keys():
            default = "(default)" if self[key] == self._default[key] else ""
            lines.append(f"    {key:{longest}} : "
                         f"{self[key].__repr__():{70-longest}}"
                         f"{default}")
        return "\n".join(lines)

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
                self._feedback,
                self._name,
                self.__doc__,
                tuple(self.keys()),
                tuple(self.values())
                )
            )
