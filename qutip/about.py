# This file is part of QuTiP.
#
#    QuTiP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    QuTiP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTiP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011-2013, Paul D. Nation & Robert J. Johansson
#
###########################################################################

"""
Module for aboutbox or command line ouput of information on QuTiP and
dependencies.
"""

import sys
import os
import numpy
import scipy
import qutip.settings
from qutip import __version__ as qutip_version


def about():
    """
    About box for qutip. Gives version numbers for
    QuTiP, NumPy, SciPy, and MatPlotLib.
    GUI version requires PySide or PyQt4.
    """
    if (qutip.settings.qutip_graphics == 'YES' and
            qutip.settings.qutip_gui != "NONE"):

        from qutip.gui import AboutBox
        import matplotlib
        if qutip.settings.qutip_gui == "PYSIDE":
            from PySide import QtGui
        elif qutip.settings.qutip_gui == "PYQT4":
            from PyQt4 import QtGui

        # checks if QApplication already exists (needed for iPython)
        app = QtGui.QApplication.instance()

        # create QApplication if it doesnt exist
        if not app:
            app = QtGui.QApplication(sys.argv)

        box = AboutBox()
        box.show()
        box.activateWindow()
        box.raise_()
        app.exec_()

    else:
        print('')
        print("QuTiP: The Quantum Toolbox in Python")
        print("Copyright (c) 2011-2013")
        print("Paul D. Nation & Robert J. Johansson")
        print('')
        print("QuTiP Version:       " + qutip.__version__)
        print("Numpy Version:       " + numpy.__version__)
        print("Scipy Version:       " + scipy.__version__)
        try:
            import matplotlib
            matplotlib_ver = matplotlib.__version__
        except:
            matplotlib_ver = 'None'
        print(("Matplotlib Version:  " + matplotlib_ver))
        print('')

        try:
            import PySide
            pyside_ver = PySide.__version__
        except:
            pyside_ver = 'None'
        try:
            import PyQt4.QtCore as qt4Core
            pyqt4_ver = qt4Core.PYQT_VERSION_STR
        except:
            pyqt4_ver = 'None'
        if sys.platform == 'darwin':
            try:
                import Foundation
                pyobjc = 'Yes'
            except:
                pyobjc = 'No'
        print(("PySide Version:      " + pyside_ver))
        print(("PyQt4 Version:       " + pyqt4_ver))
        if sys.platform == 'darwin':
            print(("PyObjc Installed:    " + pyobjc))
