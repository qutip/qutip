# Required for Sphinx to follow autodoc_type_aliases
from __future__ import annotations

from ..settings import settings
from .numpy_backend import np as qt_np
import numpy
from typing import overload, Literal, Any
import types

__all__ = ["CoreOptions"]


class QutipOptions:
    """
    Class for basic functionality for qutip's options.

    Define basic method to wrap an ``options`` dict.
    Default options are in a class _options dict.

    Options can also act as properties. The ``_properties`` map options keys to
    a function to call when the ``QutipOptions`` become the default.
    """

    _options: dict[str, Any] = {}
    _properties = {}
    _settings_name = None  # Where the default is in settings

    def __init__(self, **options):
        self.options = self._options.copy()
        for key in set(options) & set(self.options):
            self[key] = options.pop(key)
        if options:
            raise KeyError(f"Options {set(options)} are not supported.")

    def __contains__(self, key: str) -> bool:
        return key in self.options

    def __getitem__(self, key: str) -> Any:
        # Let the dict catch the KeyError
        return self.options[key]

    def __setitem__(self, key: str, value: Any) -> None:
        # Let the dict catch the KeyError
        self.options[key] = value
        if (
            key in self._properties
            and self is getattr(settings, self._settings_name)
        ):
            self._properties[key](value)

    def __repr__(self, full: bool = True) -> str:
        out = [f"<{self.__class__.__name__}("]
        for key, value in self.options.items():
            if full or value != self._options[key]:
                out += [f"    '{key}': {repr(value)},"]
        out += [")>"]
        if len(out) - 2:
            return "\n".join(out)
        else:
            return "".join(out)

    def __enter__(self):
        self._backup = getattr(settings, self._settings_name)
        self._set_as_global_default()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        exc_traceback: types.TracebackType | None,
    ) -> None:
        self._backup._set_as_global_default()

    def _set_as_global_default(self):
        setattr(settings, self._settings_name, self)
        for key in self._properties:
            self._properties[key](self.options[key])


def _set_default_dtype(new_dtype):
    import qutip
    # Catch non existing alias and unsuported types
    if new_dtype is not None:
        qutip.core.data.to.parse(new_dtype)
    for dispatcher in qutip.core.data.to.dispatchers:
        dispatcher.rebuild_lookup()


def _set_default_dtype_scope(new_range):
    import qutip
    if new_range not in ["creation", "missing", "full"]:
        raise ValueError(
            "'default_dtype_scope' must be one of "
            "'creation', 'missing', 'full'"
        )
    for dispatcher in qutip.core.data.to.dispatchers:
        dispatcher.rebuild_lookup()


class CoreOptions(QutipOptions):
    """
    Options used by the core of qutip such as the tolerance of :obj:`.Qobj`
    comparison or coefficient's format.

    Values can be changed in ``qutip.settings.core`` or by using context:

        ``with CoreOptions(atol=1e-6): ...``

    ********
    Options:
    ********

    auto_tidyup : bool
        Whether to tidyup during sparse operations.

    auto_tidyup_dims : bool [True]
        Whether to track the structure of scalar dimensions.
        Without auto_tidyup_dims:

            ``basis([2, 2]).dims == [[2, 2], [1, 1]]``

        With auto_tidyup_dims:

            ``basis([2, 2]).dims == [[2, 2], [1]]``

    atol : float {1e-12}
        General absolute tolerance. Used in various functions to round off
        small values.

    rtol : float {1e-12}
        General relative tolerance.

    auto_tidyup_atol : float {1e-14}
        The absolute tolerance used in automatic tidyup (see the
        ``auto_tidyup`` parameter above) and the default value of ``atol`` used
        in :meth:`Qobj.tidyup`.

    function_coefficient_style : str {"auto"}
        The signature expected by function coefficients. The options are:

        - "pythonic": the signature should be ``f(t, ...)`` where ``t``
          is the time and the ``...`` are the remaining arguments passed
          directly into the function. E.g. ``f(t, w, b=5)``.

        - "dict": the signature shoule be ``f(t, args)`` where ``t`` is
          the time and ``args`` is a dict containing the remaining arguments.
          E.g. ``f(t, {"w": w, "b": 5})``.

        - "auto": select automatically between the two options above based
          on the signature of the supplied function. If the function signature
          is exactly ``f(t, args)`` then ``dict`` is used. Otherwise
          ``pythonic`` is used.

    default_dtype : Nonetype, str, type {None}
        The specified data type will be used as output for different qutip
        functions determined by the "default_dtype_scope" options. Any
        data-layer known to ``qutip.data.to`` is accepted. When ``None``, these
        functions will default to a sensible data type.

    default_dtype_scope : str {"creation"}
        Control where the default_dtype apply.

        - "creation": Used in function creating :obj:`.Qobj`, such as
          :func:"qeye" or :func:"rand_herm". Also used when creating `Qobj`
          from list.

        - "missing": "default_dtype" is also used in operations between data
          layers where no exact operation match. For example ``Dense + Dense``
          will return a ``Dense``, but ``Dense + CSR`` will use the default
          type.

        - "full": "default_dtype" is used for the output of Qobj operations and
          forced when creating any Qobj. Be careful as it can affect the speed
          of operation greatly.
    """

    _options = {
        # use auto tidyup
        "auto_tidyup": True,
        # use auto tidyup dims
        "auto_tidyup_dims": True,
        # general absolute tolerance
        "atol": 1e-12,
        # general relative tolerance
        "rtol": 1e-12,
        # use auto tidyup absolute tolerance
        "auto_tidyup_atol": 1e-14,
        # signature style expected by function coefficients
        "function_coefficient_style": "auto",
        # Default Qobj dtype for Qobj create function
        "default_dtype": None,
        # Where the default_dtype apply:
        # - "creation": Used in functions creating Qobj.
        # - "missing": Missing specialisation output use default.
        # - "full": All data layer operation output that type.
        "default_dtype_scope": "creation",
        # Expect, trace, etc. will return real for hermitian matrices.
        # Hermiticity checks can be slow, stop jitting, etc.
        "auto_real_casting": True,
        # Default backend is numpy
        "numpy_backend": numpy
    }
    _settings_name = "core"
    _properties = {
        "numpy_backend": qt_np._qutip_setting_backend,
    }

    @overload
    def __getitem__(
        self,
        key: Literal["auto_tidyup", "auto_tidyup_dims", "auto_real_casting"],
    ) -> bool: ...

    @overload
    def __getitem__(
        self, key: Literal["atol", "rtol", "auto_tidyup_atol"]
    ) -> float: ...

    @overload
    def __getitem__(
        self, key: Literal["function_coefficient_style", "default_dtype_scope"]
    ) -> str: ...

    @overload
    def __getitem__(self, key: Literal["default_dtype"]) -> str | None: ...

    def __getitem__(self, key: str) -> Any:
        # Let the dict catch the KeyError
        return self.options[key]

    @overload
    def __setitem__(
        self,
        key: Literal["auto_tidyup", "auto_tidyup_dims", "auto_real_casting"],
        value: bool,
    ) -> None: ...

    @overload
    def __setitem__(
        self, key: Literal["atol", "rtol", "auto_tidyup_atol"], value: float
    ) -> None: ...

    @overload
    def __setitem__(
        self,
        key: Literal["function_coefficient_style", "default_dtype_scope"],
        value: str
    ) -> None: ...

    @overload
    def __setitem__(
        self, key: Literal["default_dtype"], value: str | None
    ) -> None: ...

    def __setitem__(self, key: str, value: Any) -> None:
        # Let the dict catch the KeyError
        super().__setitem__(key, value)


# Creating the instance of core options to use everywhere.
CoreOptions()._set_as_global_default()
# Add properties after initial setup to not trigger the lookup rebuild
CoreOptions._properties["default_dtype"] = _set_default_dtype
CoreOptions._properties["default_dtype_scope"] = _set_default_dtype_scope
