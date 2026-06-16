__all__ = []

import warnings
import weakref
import typing


class Field:
    """
    Simple class to represent the default of an options and the type of values
    it can be.

    Parameters
    ----------
    name : str
        Name of the options: "atol", "keep_runs_results", etc.
    default : Any
        Default value of the options to use when creating a new Solver or
        Integrator instance.
    type_hint : Any, optional
        Type hint to be added to the options attributes annotation.
        ``get_type_hint(solver_instance.options.atol)`` will return this value
        of the atol Field.
    validation : callable | str, optional
        Function to validate the option value. Expected signature is:
        (type_hint | Any) -> type_hint, raising error when the input type is
        wrong.
        Some generic check are available as string:
        - "int", "float",
        - ">=0", ">0", "<=0", "<0": arbitrary constrain need custom function.
        - "literal": type_hint must be a typing.Literal object
        It is possible to join checks: "int >=0".
    """

    def __init__(self, name, default, type_hint=None, validation=None):
        self.name = name
        self.default = default
        self.optional = self.default is None
        self.type_hint = type_hint
        self.validation = validation

    def update_default(self, new_default):
        self.default = self.validate(new_default)

    def validate(self, val):
        if self.optional and val is None:
            return val

        if callable(self.validation):
            return self.validation(val)

        if self.validation == "literal":
            if val not in typing.get_args(self.type_hint):
                raise TypeError(
                    f"{self.name} can only that the values "
                    f"{typing.get_args(self.type_hint)}"
                )
        if "int" in self.validation:
            if int(val) != val:
                raise TypeError(f"Expected an int, got {type(val)}")
            val = int(val)
        elif "float" in self.validation:
            if float(val) != val:
                raise TypeError(f"Expected an float, got {type(val)}")
            val = float(val)
        if ">=0" in self.validation:
            if val < 0:
                raise TypeError(f"Expected a positive number")
        elif ">0" in self.validation:
            if val <= 0:
                raise TypeError(f"Expected a positive, non-zero number")
        if "<=0" in self.validation:
            if val > 0:
                raise TypeError(f"Expected a negative number")
        if "<0" in self.validation:
            if val >= 0:
                raise TypeError(f"Expected a negative, non-zero number")
        return val


class _OptionsDefault:
    def __init__(self, default):
        super().__setattr__("_default", {})
        self.__annotation__ = {}
        for key, val in default:
            if not isinstance(val, Field):
                if not isinstance(val, tuple):
                    val = val,
                self._default[key] = Field(key, *val)
            self.__annotation__[key] = self._default[key].type_hint
            super().__setattr__(key,  self._default[key])

    def __setitem__(self, key, val):
        if key not in self._default:
            raise KeyError
        self._default[key].update_default(val)

    def __getitem__(self, key):
        return self._default[key].default

    def __getattribute__(self, key):
        if key in super().__getattribute__("_default"):
            return self._default[key]
        return super().__getattribute__(key)

    def __setattr__(self, key, val):
        if key in self._default:
            self._default[key].update_default(val)
        super().__setattr__(key, val)


class _SolverOptions():
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
        self._dict = dict(**{**self._default, **kwargs})

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, val):
        if key not in self._default:
            raise KeyError(f"Options {key} is not supported.")
        if val is None:
            val = self._default[key]
        if val == self._dict[key]:
            return
        self._dict[key] = val
        if (updater := self._on_update()) is not None:
            updater(key)

    def __delitem__(self, key):
        if key not in self._default:
            raise KeyError(f"Options {key} is not supported.")
        self._dict[key] = self._default[key]
        if (updater := self._on_update()) is not None:
            updater(key)

    def copy(self):
        return self.__class__(
            self._default,
            None,
            self._name,
            self.__doc__,
            **self._dict
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
