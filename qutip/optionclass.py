from qutip.settings import settings


def optionclass(name, parent=settings):
    """Make the class an Options object of Qutip and register the object
    default as qutip.settings."name".

    Add the methods:
        __init__:
            Allow to create from data in files or from default with attributes
            overwritten by keywords.
            Properties with setter can also be set as kwargs.
        save(file), load(file), reset():
            Save, load, reset will affect all attributes that can be saved
            as defined in qutip.configrc.getter.
        __repr__():
            Make a clean print of all attribute and properties.
    and the attributes:
        _name
        _fullname
        _types
        _isDefault
        _defaultInstance

    * Any attribute starting with "_" are excluded.

    Usage:
        ``
        @optionclass(name)
        class Options:
            ...
        ``
    or
        ``
        @optionclass
        class Options
            ...
        ``
    * default name is `Options.__name__`
    """
    # The real work is in _QtOptionMaker
    if isinstance(name, str):
        # Called as
        # @QtOptionClass(name)
        # class Options:
        return _QtOptionMaker(name, parent)
    else:
        # Called as
        # @QtOptionClass
        # class Options:
        return _QtOptionMaker(name.__name__, parent)(name)


class _QtOptionMaker:
    def __init__(self, name, parent):
        self.name = name
        self.parent = parent

    def __call__(self, cls):
        if hasattr(cls, "_isDefault"):
            # Already a QtOptionClass
            return

        cls.childs = []
        if not hasattr(cls, "read_only_options"):
            cls.read_only_options = {}
        cls._types = {key: type(val) for key, val in cls.options.items()}
        # Name in settings and in files
        cls._name = self.name
        # Name when printing
        cls._fullname = ".".join([self.parent._fullname, self.name])
        # Is this instance the default for the other.
        cls._isDefault = False
        # Build the default instance
        # Do it before setting __init__ since it use this default
        self._make_default(cls)

        cls.__init__ = _make_init(cls.options)
        cls.__repr__ = _qoc_repr
        cls.__getitem__ = _qoc_getitem
        cls.__setitem__ = _qoc_setitem
        cls.reset = _qoc_reset
        cls.save = _qoc_save
        cls.load = _qoc_load
        cls._all_childs = _qoc_all_childs

        return cls

    def _make_default(self, cls):
        default = cls()
        default.options = cls.options.copy()
        default._isDefault = True
        default.childs = []
        self.parent._defaultInstance.childs.append(default)
        setattr(self.parent._defaultInstance, self.name, default)
        cls._defaultInstance = default


def _make_init(all_set):
    attributes_kw = ",\n             ".join(["{}=None".format(key)
                    for key in all_set])
    attributes_set = "".join(["    if {0} is not None:\n"
                              "        self.options['{0}'] = {0}\n".format(key)
                     for key in all_set])
    code = f"""
def __init__(self, file='', *,
             {attributes_kw}):
    self.options = self._defaultInstance.options.copy()
{attributes_set}
    if file:
        self.load(file)
    for child in self._defaultInstance.childs:
        setattr(self, child._name, child.__class__())
"""
    ns = {}
    exec(code, globals(), ns)
    return ns["__init__"]


def _qoc_repr(self, _recursive=False):
    out = self._fullname + ":\n"
    longest = max(len(key) for key in self.options)
    if self.read_only_options:
        longest_readonly = max(len(key) for key in self.read_only_options)
        longest = max((longest, longest_readonly))
    for key, val in self.options.items():
        if isinstance(val, str):
            out += "{:{width}} : '{}'\n".format(key, val,
                                                width=longest)
        else:
            out += "{:{width}} : {}\n".format(key, val,
                                              width=longest)
    for key, val in self.read_only_options.items():
        out += "{:{width}} : {}\n".format(key, val,
                                          width=longest)
    out += "\n"
    if _recursive:
         out += "".join([child.__repr__(_recursive)
                         for child in self.childs])
    return out


def _qoc_reset(self, _recursive=False):
    """Reset instance to the default value or the default to Qutip's default"""
    if self._isDefault:
        self.options = self.__class__.options.copy()
        if _recursive:
            for child in self.childs:
                child.reset()
    else:
        self.options = self._defaultInstance.options.copy()


def _qoc_all_childs(self):
    optcls = [self]
    for child in self.childs:
         optcls += child._all_childs()
    return optcls


def _qoc_save(self, file="qutiprc", _recursive=False):
    """Save to desired file. 'qutiprc' if not specified"""
    import qutip.configrc as qrc
    if _recursive:
        optcls = self._all_childs()
    else:
        optcls = [self]
    qrc.write_rc_object(file, optcls)


def _qoc_load(self, file="qutiprc", _recursive=False):
    """Load from desired file. 'qutiprc' if not specified"""
    import qutip.configrc as qrc
    if _recursive:
        optcls = self._all_childs()
    else:
        optcls = [self]
    qrc.load_rc_object(file, optcls)


def _qoc_getitem(self, key):
    if key in self.read_only_options:
        return self.read_only_options[key]
    return self.options[key]


def _qoc_setitem(self, key, value):
    if key in self.read_only_options:
        raise KeyError("Read-only value")
    self.options[key] = value
