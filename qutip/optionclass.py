import qutip.settings as qset

def optionclass(name):
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
        _all
        _repr_keys
        _name
        _fullname
        _isDefault
        _defaultInstance

    * Any attribute starting with "_" are excluded.

    Usage:
        ``
        @QtOptionClass(name)
        class Options:
            ...
        ``
    or
        ``
        @QtOptionClass
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
        return _QtOptionMaker(name)
    else:
        # Called as
        # @QtOptionClass
        # class Options:
        return _QtOptionMaker(name.__name__)(name)


class _QtOptionMaker:
    def __init__(self, name):
        self.name = name

    def __call__(self, cls):
        if hasattr(cls, "_isDefault"):
            # Already a QtOptionClass
            if self.name not in __self:
                self._make_default(cls)
            return

        # attributes that to be saved
        cls._all = [key for key in cls.__dict__
                    if self._valid(cls, key)]
        # attributes to print
        cls._repr_keys = [key for key in cls.__dict__
                          if self._valid(cls, key, _repr=True)]
        # Name in settings and in files
        cls._name = self.name
        # Name when printing
        cls._fullname = ".".join([cls.__module__, cls.__name__])
        # Is this instance the default for the other.
        cls._isDefault = False
        # Build the default instance
        # Do it before setting __init__ since it use this default
        self._make_default(cls)

        # add methods
        # __init__ is dynamically build to get a meaningfusl signature
        _all_set = [key for key in cls.__dict__
                    if self._valid(cls, key, _set=True)]
        attributes_kw = ",\n             ".join(["{}=None".format(var)
                        for var in _all_set])
        attributes_set = "".join(["\n    self.{0} = {0} if {0} is not None "
                         "else self._defaultInstance.{0}".format(var)
                         for var in _all_set])
        code = f"""
def __init__(self, file='', *,
             {attributes_kw}):
    {attributes_set}
    if file:
        self.load(file)
"""
        ns = {}
        exec(code, globals(), ns)
        cls.__init__ = ns["__init__"]
        cls.__repr__ = _qoc_repr_
        cls.reset = _qoc_reset
        cls.save = _qoc_save
        cls.load = _qoc_load
        return cls

    @staticmethod
    def _valid(cls, key, _repr=False, _set=False):
        # Can it be saved, printed, initialed?
        import qutip.configrc as qrc
        if key.startswith("_"):
            return False
        data = getattr(cls, key)
        if _repr and isinstance(data, property):
            # Print all properties
            return True
        if _set and isinstance(data, property) and data.fset is not None:
            # Properties with a setter can be set in __init__
            return True
        # Only these types can be saved
        return type(data) in qrc.getter

    def _make_default(self, cls):
        import qutip.configrc as qrc
        default = cls()
        for key in cls._all:
            default.__dict__[key] = cls.__dict__[key]
        default._isDefault = True
        default._fullname = "qutip.settings." + self.name
        setattr(qset, self.name, default)
        cls._defaultInstance = default
        qrc.sections.append((self.name, default))


def _qoc_repr_(self):
    out = self._fullname + ":\n"
    longest = max(len(key) for key in self._repr_keys)
    for key in self._repr_keys:
        out += "{:{width}} : {}\n".format(key, getattr(self, key),
                                          width=longest)
    return out


def _qoc_reset(self):
    """Reset instance to the default value or the default to Qutip's default"""
    if self._isDefault:
        [setattr(self, key, getattr(self.__class__, key))
         for key in self._all]
    else:
        [setattr(self, key, getattr(self._defaultInstance, key))
         for key in self._all]


def _qoc_save(self, file="qutiprc"):
    """Save to desired file. 'qutiprc' if not specified"""
    import qutip.configrc as qrc
    qrc.write_rc_object(file, self._name, self)


def _qoc_load(self, file="qutiprc"):
    """Load from desired file. 'qutiprc' if not specified"""
    import qutip.configrc as qrc
    qrc.load_rc_object(file, self._name, self)
