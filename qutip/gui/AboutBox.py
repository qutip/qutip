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

import sys,os
from urllib2 import urlopen
if os.environ['QUTIP_GUI']=="PYSIDE":
    from PySide import QtGui, QtCore

elif os.environ['QUTIP_GUI']=="PYQT4":
    from PyQt4 import QtGui, QtCore
   
import numpy,scipy,matplotlib
CD_BASE = os.path.dirname(__file__)

class AboutBox(QtGui.QWidget):
    def __init__(self, Qversion, parent=None):
        QtGui.QWidget.__init__(self, parent)
        #WINDOW PROPERTIES
        self.setWindowTitle('About QuTiP')
        self.resize(430, 450)
        self.center()
        self.setFocus()
        #self.setAttribute(Qt.WA_TranslucentBackground)#transparent
        #self.setWindowOpacity(0.95)
        #self.setWindowFlags(Qt.Popup)#no titlebar
        #IMAGES--------------------
        logo=QtGui.QLabel(self)
        logo.setGeometry((self.width()-200)/2, 0, 200, 163)
        logo.setPixmap(QtGui.QPixmap(CD_BASE + "/logo.png"))
        #TEXT--------------------
        tlabel = QtGui.QLabel(self)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        if sys.platform=='darwin':
            font.setPointSize(20)
        else:
            font.setPointSize(15)
        fm = QtGui.QFontMetrics(font)
        tstring="QuTiP: The Quantum Toolbox in Python"
        pixelswide = fm.width(tstring)
        tlabel.setFont(font)
        tlabel.setText(tstring)
        tlabel.move((self.width()-pixelswide)/2.0, 170)
        #
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
        label = QtGui.QLabel(self)
        font.setFamily("Arial")
        font.setBold(False)
        if sys.platform=='darwin':
            font.setPointSize(14)
        else:
            font.setPointSize(11)
        label.setFont(font)
        fm = QtGui.QFontMetrics(font)
        #check for updated version
        try:
            current = urlopen("http://qutip.googlecode.com/svn/doc/current_version.txt").read()
            current=current.replace('.','')[0:3]
        except:
            current=None
        if sys.platform!='darwin':
            lstring="QuTiP Version:           "+Qversion
            pixelswide = fm.width(lstring)
            label.setText(lstring)
            label.move((self.width()-pixelswide)/2,210)
            if current and int(current)>int(Qversion.replace('.','')[0:3]):
                label2= QtGui.QLabel(self)
                label2.setFont(font)
                label2.setOpenExternalLinks(True)
                lstring2=" (<a href=http://code.google.com/p/qutip/downloads/list>Get New</a>)"+"\n"
                label2.setText(lstring2)
                label2.move((2*self.width()-pixelswide)/2,210)
            label3= QtGui.QLabel(self)
            label3.setFont(font)
            lstring3="\n"
            lstring3+="NumPy Version:         "+str(numpy.__version__)+"\n"
            lstring3+="SciPy Version:            "+str(scipy.__version__)+"\n"
            lstring3+="MatPlotLib Version:    "+str(matplotlib.__version__)+"\n\n"
            lstring3+="PySide Version:         "+str(pyside_ver)+"\n"
            lstring3+="PyQt4 Version:           "+str(pyqt4_ver)
            label3.setText(lstring3)
            label3.move((self.width()-pixelswide)/2,210)
        else:
            lstring="QuTiP Version:           "+Qversion
            pixelswide = fm.width(lstring)
            label.setText(lstring)
            label.move((self.width()-pixelswide)/2,210)
            if current and int(current)>int(Qversion.replace('.','')[0:3]):
                label2= QtGui.QLabel(self)
                label2.setFont(font)
                label2.setOpenExternalLinks(True)
                lstring2=" (<a href=http://code.google.com/p/qutip/downloads/list>Get New</a>)"+"\n"
                label2.setText(lstring2)
                label2.move((2*self.width()-pixelswide)/2,210)
            label3= QtGui.QLabel(self)
            label3.setFont(font)
            lstring3="\n"
            lstring3+="NumPy Version:         "+str(numpy.__version__)+"\n"
            lstring3+="SciPy Version:            "+str(scipy.__version__)+"\n"
            lstring3+="MatPlotLib Version:    "+str(matplotlib.__version__)+"\n\n"
            lstring3+="PySide Version:         "+str(pyside_ver)+"\n"
            lstring3+="PyQt4 Version:           "+str(pyqt4_ver)+"\n"
            lstring3+="PyObjc Installed:        "+str(pyobjc)
            label3.setText(lstring3)
            label3.move((self.width()-pixelswide)/2,210)
        #
        alabel = QtGui.QLabel(self)
        astring="Copyright (c) 2011-2012, P. D. Nation & R. J. Johansson"
        pixelswide = fm.width(astring)
        alabel.setFont(font)
        alabel.setText(astring)
        alabel.move((self.width()-pixelswide)/2, 350)
        #
        clabel = QtGui.QLabel(self)
        clabel.setFont(font)
        clabel.setText("QuTiP is released under the GPL3.\n"
                        +"See the enclosed COPYING.txt\nfile for more information.")
        clabel.move((self.width()-pixelswide)/2, 380)
        #BUTTONS-----------------
        quit = QtGui.QPushButton('Close', self)
        quit.setFont(font)
        quit.setGeometry((self.width()-90), 395, 80, 40)
        #quit.setFocusPolicy(QtCore.Qt.NoFocus)
        quit.clicked.connect(self.close)
    def center(self):
        screen = QtGui.QDesktopWidget().screenGeometry()
        size =  self.geometry()
        self.move((screen.width()-size.width())/2, (screen.height()-size.height())/2)


    
