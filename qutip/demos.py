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
# Copyright (C) 2011, Paul D. Nation & Robert J. Johansson
#
###########################################################################
import sys,os
import examples
def demos():
    if os.environ['QUTIP_GRAPHICS']=='YES':
        from gui import Examples,exconfig
        if os.environ['QUTIP_GUI']=="PYSIDE":
            from PySide import QtGui, QtCore
        elif os.environ['QUTIP_GUI']=="PYQT4":
            from PyQt4 import QtGui, QtCore
        def start_gui():
            app=QtGui.QApplication.instance()#checks if QApplication already exists (needed for iPython)
            if not app:#create QApplication if it doesnt exist
                app = QtGui.QApplication(sys.argv)
            gui=Examples()
            gui.show()
            gui.raise_()
            app.exec_()
        while exconfig.option<123456:
            exconfig.option=123456
            start_gui()
            if exconfig.option==1:
                examples.qobjbasics()
            elif exconfig.option==2:
                examples.wignercat()
            elif exconfig.option==5:
                examples.trilinearmc()
            
            
            