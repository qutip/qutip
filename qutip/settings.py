"""
This module contains settings for the QuTiP graphics, multiprocessing, and
tidyup functionality, etc.
"""
from __future__ import absolute_import
import qutip.configrc as qrc


class Settings:
    """
    Qutip default settings and options.
    `print(qutip.settings)` to list all available options.
    `help(qutip.settings.solver)` will explain the use of each options
        in `solver`.

    """
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

    def reset(self):
        """
        Reset all options to qutip's defaults.
        """
        for child in self._children:
            child.reset(True)

    def save(self, file="qutiprc"):
        """
        Save the default in a file in '$HOME/.qutip/'.
        Use full path to same elsewhere.
        The file 'qutiprc' is loaded when importing qutip.
        """
        optcls = self._all_children()
        qrc.write_rc_object(file, optcls)

    def load(self, file="qutiprc"):
        """
        Load the default in a file in '$HOME/.qutip/'.
        Use full path to same elsewhere.
        """
        optcls = self._all_children()
        qrc.load_rc_object(file, optcls)

    def __repr__(self):
        return "".join(child.__repr__(True) for child in self._children)


settings = Settings()
