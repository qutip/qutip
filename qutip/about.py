#This file is part of QuTIP.
#
#    QuTIP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#    QuTIP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTIP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson
#
###########################################################################

"""
Module for aboutbox or command line ouput of information on QuTiP and dependencies.
"""

import sys,os
import numpy,scipy
import qutip.settings

CD_BASE = os.path.dirname(__file__) # get directory of about.py file
#execfile(os.path.join(CD_BASE, "_version.py")) #execute _version.py file in CD_BASE directory
exec(compile(open(os.path.join(CD_BASE, "_version.py")).read(), os.path.join(CD_BASE, "_version.py"), 'exec'))
def about():
    """
    About box for qutip. Gives version numbers for 
    QuTiP, NumPy, SciPy, and MatPlotLib.
    GUI version requires PySide or PyQt4.
    """
    if qutip.settings.qutip_graphics=='YES' and qutip.settings.qutip_gui!="NONE":
        from gui import AboutBox
        import matplotlib
        if qutip.settings.qutip_gui=="PYSIDE":
            from PySide import QtGui, QtCore
        elif qutip.settings.qutip_gui=="PYQT4":
            from PyQt4 import QtGui, QtCore
        
        app=QtGui.QApplication.instance()#checks if QApplication already exists (needed for iPython)
        if not app:#create QApplication if it doesnt exist
            app = QtGui.QApplication(sys.argv)
        box=AboutBox(version)
        box.show()
        box.activateWindow()
        box.raise_()
        app.exec_()
        
        
        
    else:
        print('')
        print("QuTiP: The Quantum Toolbox in Python")
        print("Copyright (c) 2011-2012")
        print("Paul D. Nation & Robert J. Johansson")
        print('')
        print("QuTiP Version:       "+version)
        print("Numpy Version:       "+numpy.__version__)
        print("Scipy Version:       "+scipy.__version__)
        try:
            import matplotlib
            matplotlib_ver = matplotlib.__version__
        except:
            matplotlib_ver = 'None' 
        print("Matplotlib Version:  " + matplotlib_ver)
        print('')

        try:
            import PySide
            pyside_ver=PySide.__version__
        except:
            pyside_ver='None'
        try:
            import PyQt4.QtCore as qt4Core
            pyqt4_ver=qt4Core.PYQT_VERSION_STR
        except:
            pyqt4_ver='None'
        if sys.platform=='darwin':
            try:
                import Foundation
                pyobjc='Yes'
            except:
                pyobjc='No'
        print("PySide Version:      "+pyside_ver)
        print("PyQt4 Version:       "+pyqt4_ver)
        if sys.platform=='darwin':
            print("PyObjc Installed:    "+pyobjc)


