"""

"""

from qutip.settings import settings

def store_options(options, name, file):
    pass


def save_options_file(options, file):
    pass


def read_file(file, name=None):
    if file[-3:] != ".py":
        file = os.path.join(settings.tmproot, 'qutip_saved_options.py')

    spec = importlib.util.spec_from_file_location('qutipoptions', file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    #mod = importlib.import_module(file_name)
    if name:
        return getattr(module, name)
    return module


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
        if key in cls.check:
            value = cls.check[key](value)
        cls.default[key] = val


class QutipOptions(metaclass=MetaOptions):
    default = {}
    check = {}
    name = "base_options"

    def __init__(self, base=None, *, _strick=True, **options):
        if isinstance(base, dict):
            options.update(base)

        elif isinstance(base, QutipOptions):
            options.update(base.options)

        if _strick and (set(options) - set(self.default)):
            raise KeyError("Unknown option(s): " +
                           f"{set(options) - set(self.default)}")
        self.options = self.default.copy()
        self._from_dict(options)

    def copy(self):
        return self.__class__(self)

    def __contains__(self, key):
        return key in self.options

    def __getitem__(self, key):
        # Let the dict catch the KeyError
        return self.options[key]

    def __setitem__(self, key, value):
        # Let the dict catch the KeyError
        if key in self.check:
            value = self.check[key](value)
        self.options[key] = value

    def __repr__(self):
        out = "{\n"
        for key, value in self.options.items():
            out += f"    '{key}' : {repr(value)},\n"
        out += "}\n"
        return out

    def __str__(self):
        out = self.name + ":\n"
        longest = max(len(key) for key in self.options)
        for key, val in self.options.items():
            if isinstance(val, str):
                out += "{:{width}} : '{}'\n".format(key, val, width=longest)
            else:
                out += "{:{width}} : {}\n".format(key, val, width=longest)
        out += "\n"
        return out

    def set_has_default(self):
        for key in self.options:
            self.__class__[key] = self.options[key]

    def _from_dict(self, opt):
        for key in set(opt) & set(self.options):
            self[key] = opt[key]

    def save(self, file=None):
        if file[-3:] != ".py":
            name = file
            file = os.path.join(settings.tmproot, 'saved_options.py')
            store_options(self, name, file)
        else:
            save_options_file(self, name, file)

    def __enter__(self):
        self.__backup_default = self.__class__.default
        self.__class__.default = self.options

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.__class__.default = self.__backup_default
