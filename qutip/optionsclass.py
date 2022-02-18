"""

"""

from qutip.settings import settings


class MetaOptions(type):
    """
    Allow to define __getitem__ and __setitem__ as classmethod:
    >>> Options[key] = val
    This change the default of all new instance of options.
    >>> opt = Options()
    >>> opt[key] = val
    This change the value for only this instance.
    """
    def __getitem__(cls, key):
        return cls.default[key]

    def __setitem__(cls, key, val):
        if key not in cls.default:
            raise KeyError(f"Key '{key}' is not supported.")
        if key in cls.check:
            value = cls.check[key](value)
        cls.default[key] = val


class QutipOptions(metaclass=MetaOptions):
    default = {}
    check = {}

    def __init__(self, base=None, *, _strict=True, _frozen=False, **options):
        if isinstance(base, dict):
            options.update(base)

        elif isinstance(base, QutipOptions):
            options.update(base.options)

        self.frozen = _frozen
        self.options = self.default.copy()
        self._from_dict(options)
        if _strict and options:
            raise KeyError(f"Unknown option(s): {set(options)}")

    def copy(self):
        return self.__class__(self)

    def __contains__(self, key):
        return key in self.options

    def __getitem__(self, key):
        # Let the dict catch the KeyError
        return self.options[key]

    def __setitem__(self, key, value):
        # Let the dict catch the KeyError
        if self.frozen:
            raise RuntimeError("Options associated cannot be modified, "
                               "only overwritten.")
        if key not in self.options:
            raise KeyError(f"Key '{key}' is not supported.")
        if key in self.check:
            value = self.check[key](value)
        self.options[key] = value

    def __repr__(self):
        out = type(self).__name__ + "({\n"
        for key, value in self.options.items():
            out += f"    '{key}' : {repr(value)},\n"
        out += "})\n"
        return out

    def __str__(self):
        longest = max(len(key) for key in self.options)
        out = type(self).__name__ + ":\n"
        for key, val in self.options.items():
            if isinstance(val, str):
                out += f"    {key:{longest}} : '{val}'\n"
            else:
                out += f"    {key:{longest}} : {val}\n"
        out += "\n"
        return out

    def _from_dict(self, opt):
        for key in set(opt) & set(self.options):
            self[key] = opt.pop(key)

    def __enter__(self):
        self.__backup_default = self.__class__.default
        self.__class__.default = self.options

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.__class__.default = self.__backup_default
