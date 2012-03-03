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
    def __init__(self,Qversion): 
        QtGui.QWidget.__init__(self) 
         
        self.setGeometry(0,0, 360,480) 
        self.setWindowTitle("About QuTiP") 
        self.setWindowIcon(QtGui.QIcon(CD_BASE + "/logo.png")) 
        self.resize(360,480) 
        self.setMinimumSize(360,480) 
        self.center() 
        self.setFocus()
        
        logo=QtGui.QLabel(self)
        logo.setGeometry((self.width()-200)/2, 0, 200, 163)
        logo.setPixmap(QtGui.QPixmap(CD_BASE + "/logo.png"))
        
        tlabel = QtGui.QLabel(self)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        if sys.platform=='darwin':
            font.setPointSize(17)
        else:
            font.setPointSize(15)
        fm = QtGui.QFontMetrics(font)
        tstring="QuTiP: The Quantum Toolbox in Python"
        pixelswide = fm.width(tstring)
        tlabel.setFont(font)
        tlabel.setText(tstring)
        tlabel.move((self.width()-pixelswide)/2.0, 165)
        
        #first tab text
        tab_widget = QtGui.QTabWidget(self) 
        tab_widget.move(10,200)
        tab_widget.resize(340,220)
        tab1 = QtGui.QWidget(self) 
        tab1_vert = QtGui.QVBoxLayout(tab1) 
        tab_widget.addTab(tab1, "Version Info")
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
        
        #check for updated version
        try:
            current = urlopen("http://qutip.googlecode.com/svn/doc/current_version.txt").read()
        except:
            current=None
        tab1_font = QtGui.QFont()
        tab1_font.setFamily("Arial")
        tab1_font.setBold(False)
        if sys.platform=='darwin':
            tab1_font.setPointSize(15)
        else:
            tab1_font.setPointSize(13)
        fm = QtGui.QFontMetrics(tab1_font)
        label = QtGui.QLabel(self)
        label.setFont(tab1_font)
        if sys.platform!='darwin':
            lstring="QuTiP Version:           "+Qversion
            pixelswide = fm.width(lstring)
            label.setText(lstring)
            tab1_vert.addWidget(label)
            if current or int(current.replace('.','')[0:3])>int(Qversion.replace('.','')[0:3]):
                label.setOpenExternalLinks(True)
                lstring+=" (<a href=http://code.google.com/p/qutip/downloads/list>Update</a>)"+"\n"
            label.setText(lstring)
            tab1_vert.addWidget(label)
            label3= QtGui.QLabel(tab1)
            label3.setFont(tab1_font)
            lstring3="\n"
            lstring3+="NumPy Version:         "+str(numpy.__version__)+"\n"
            lstring3+="SciPy Version:            "+str(scipy.__version__)+"\n"
            lstring3+="MatPlotLib Version:    "+str(matplotlib.__version__)+"\n\n"
            lstring3+="PySide Version:         "+str(pyside_ver)+"\n"
            lstring3+="PyQt4 Version:           "+str(pyqt4_ver)
            label3.setText(lstring3)
        else:
            lstring="QuTiP Version:           "+Qversion
            pixelswide = fm.width(lstring)
            label.setText(lstring)
            tab1_vert.addWidget(label)
            if current and int(current.replace('.','')[0:3])>int(Qversion.replace('.','')[0:3]):
                label.setOpenExternalLinks(True)
                lstring+=" (<a href=http://code.google.com/p/qutip/downloads/list>Update</a>)"+"\n"
            label.setText(lstring)
            tab1_vert.addWidget(label)
            label3= QtGui.QLabel(tab1)
            label3.setFont(tab1_font)
            lstring3="\n"
            lstring3+="NumPy Version:         "+str(numpy.__version__)+"\n"
            lstring3+="SciPy Version:            "+str(scipy.__version__)+"\n"
            lstring3+="MatPlotLib Version:    "+str(matplotlib.__version__)+"\n\n"
            lstring3+="PySide Version:         "+str(pyside_ver)+"\n"
            lstring3+="PyQt4 Version:           "+str(pyqt4_ver)+"\n"
            lstring3+="PyObjc Installed:        "+str(pyobjc)
            label3.setText(lstring3)
        
        
        tab1_vert.addWidget(label3)
        dev_text=QtGui.QLabel()
        dev_string="Lead Developers:Paul D. Nation, Robert J. Johansson\n\n"
        pixelswide = fm.width(dev_string)

        #tab2 text
        #dev_text.setFont(p1_font)
        dev_text.setText(dev_string)
        tab2 = QtGui.QWidget()
        tab_widget.addTab(tab2, "Developers")
        t2_ver = QtGui.QVBoxLayout(tab2)
        t2_ver.addWidget(dev_text)
        
        p1_font = QtGui.QFont()
        p1_font.setFamily("Arial")
        p1_font.setBold(False)
        p1_font.setPointSize(14)
        alabel = QtGui.QLabel(self)
        astring="Copyright (c) 2011-2012,\nP. D. Nation & J. R. Johansson"
        pixelswide = fm.width(astring)
        alabel.setFont(p1_font)
        alabel.setText(astring)
        alabel.move(10,430)
        
        
        #QUIT BUTTON-----------------
        quit = QtGui.QPushButton('Close', self)
        font.setBold(False)
        quit.setFont(font)
        quit.setGeometry((self.width()-85), 430, 80, 40)
        quit.clicked.connect(self.close)
       
     
     
    def center(self): 
        screen = QtGui.QDesktopWidget().screenGeometry() 
        size = self.frameSize()
        self.move((screen.width()-size.width())/2, (screen.height()-size.height())/2) 

if __name__=="__main__":
    app = QtGui.QApplication(sys.argv) 
    frame = AboutBox('2.0') 
    frame.show()
    frame.raise_()  
    app.exec_()


