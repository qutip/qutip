

class AllOptions:
    """
    Class that serve as the parent of all other options class.
    """
    # Temporary patch to make changes in solver's options minimal.
    # TODO: Follow up PR based on #1812 should replace this.
    def __init__(self):
        self._isDefault = True
        self._children = []
        self._fullname = "qutip.settings"
        self._defaultInstance = self

    def _all_children(self):
        optcls = []
        for child in self._children:
            optcls += child._all_children()
        return optcls


alloptions = AllOptions()


def optionsclass(name, parent=alloptions):
    """
    Use as a decorator to register the options class to `qutip.settings`.
    The default will be in added to qutip.setting.[parent].name.

    The options class should contain an `options` dictionary containing the
    option and their default value. Readonly settings can be defined in
    `read_only_options` dict.

    ```
    >>> import qutip
    >>> @qutip.optionsclass("myopt", parent=qutip.settings.solver)
    >>> class MyOpt():
    >>>     options = {"opt1": True}
    >>>     read_only_options = {"hidden": 1}

    >>> qutip.settings.solver.myopt['opt1'] = False
    >>> qutip.settings.solver.myopt['hidden']
    1

    >>> qutip.settings.solver.myopt['hidden'] = 2
    KeyError: "'hidden': Read-only value"

    >>> print(qutip.settings.solver.myopt)
    qutip.settings.solver.myopt:
    opt1   : False
    hidden : 1

    print(MyOpt(opt1=2))
    qutip.settings.solver.myopt:
    opt1   : 2
    hidden : 1
    ```

    It add the methods:
        __init__(file=None, *, option=None... )
            Allow to create from data in files or from defaults with attributes
            overwritten by keywords.
        __repr__():
            Make a clean print for all 'options' and 'read_only_options'.
        __getitem__(), __setitem__():
            Pass through to 'self.options' and 'self.read_only_options'.
            Option in 'self.read_only_options' can not be set.
        save(file='qutiprc'):
            Save the object in a file. 'qutiprc' file is loaded as default when
            loading qutip.
        load(file):
            Overwrite with options previously saved.
            Loaded options will have the same type as the default from one of:
                (bool, int, float, complex, str, object)
        reset():
            If used on an instance, reset to the default in qutip.settings.
            If used from qutip.settings..., go back to qutip's defaults.

    """
    # The real work is in _QtOptionsMaker
    if isinstance(name, str):
        # Called as
        # @QtOptionsClass(name)
        # class Options:
        return _QtOptionsMaker(name, parent)
    else:
        # Called as
        # @QtOptionsClass
        # class Options:
        return _QtOptionsMaker(name.__name__, parent)(name)


class _QtOptionsMaker:
    """
    Apply the `optionsclass` decorator.
    """
    def __init__(self, name, parent):
        self.name = name
        self.parent = parent

    def __call__(self, cls):
        if hasattr(cls, "_isDefault"):
            # Already a QtOptionsClass
            return

        if not hasattr(cls, "read_only_options"):
            cls.read_only_options = {}
        # type to used when loading from file.
        cls._types = {key: type(val) for key, val in cls.options.items()}
        # Name in settings and in files
        cls._name = self.name
        # Name when printing
        cls._fullname = ".".join([self.parent._fullname, self.name])
        # Is this instance the default for the other.
        cls._isDefault = False
        # Children in the settings tree
        cls._children = []
        # Build the default instance
        # Do it before setting __init__ since it use this default
        self._make_default(cls)

        cls.__init__ = _make_init(cls.options)
        cls.__repr__ = _repr
        cls.__getitem__ = _getitem
        cls.__setitem__ = _setitem
        cls.__contains__ = _contains
        cls.reset = _reset
        cls.save = _save
        cls.load = _load
        cls._all_children = _all_children

        return cls

    def _make_default(self, cls):
        """ Create the default and add it to the parent.
        """
        default = cls()
        default.options = cls.options.copy()
        default._isDefault = True
        default._children = []
        # The parent has the child twice: attribute and in a list.
        self.parent._defaultInstance._children.append(default)
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
             {attributes_kw},
             _child_instance=False,
             **kwargs):
    self.options = self._defaultInstance.options.copy()
{attributes_set}
    if file:
        self.load(file)
    if hasattr(self, "extra_options"):
        other_kw = dict()
        for kw in kwargs:
            if kw in self.extra_options:
                self.options[kw] = kwargs[kw]
            else:
                other_kw[kw] = kwargs[kw]
        kwargs = other_kw
    self._children = []
    for child in self._defaultInstance._children:
        self._children.append(child.__class__(file, _child_instance=True,
                                              **kwargs))
        setattr(self, child._name, self._children[-1])
        kwargs = self._children[-1].left_over_kwargs
    if _child_instance:
        self.left_over_kwargs = kwargs
    elif kwargs:
        raise KeyError("Unknown options: " +
                       " ,".join(str(key) for key in kwargs))
"""
    ns = {}
    exec(code, globals(), ns)
    return ns["__init__"]


def _repr(self, _recursive=True):
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
                         for child in self._children])
    return out


def _reset(self, _recursive=False):
    """Reset instance to the default value or the default to Qutip's default"""
    if self._isDefault:
        self.options = self.__class__.options.copy()
        if _recursive:
            for child in self._children:
                child.reset()
    else:
        self.options = self._defaultInstance.options.copy()


def _all_children(self):
    optcls = [self]
    for child in self._children:
         optcls += child._all_children()
    return optcls


def _save(self, file="qutiprc", _recursive=False):
    """Save to desired file. 'qutiprc' if not specified"""
    import qutip.configrc as qrc
    if _recursive:
        optcls = self._all_children()
    else:
        optcls = [self]
    qrc.write_rc_object(file, optcls)


def _load(self, file="qutiprc", _recursive=False):
    """Load from desired file. 'qutiprc' if not specified"""
    import qutip.configrc as qrc
    if _recursive:
        optcls = self._all_children()
    else:
        optcls = [self]
    qrc.load_rc_object(file, optcls)


def _contains(self, key):
    return key in self.read_only_options or key in self.options


def _getitem(self, key):
    # Let the dict catch the KeyError
    if key in self.read_only_options:
        return self.read_only_options[key]
    return self.options[key]


def _setitem(self, key, value):
    # Let the dict catch the KeyError
    if key in self.read_only_options:
        raise KeyError(f"'{key}': Read-only value")
    self.options[key] = value
