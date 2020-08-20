# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################
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
