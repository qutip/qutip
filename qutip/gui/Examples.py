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
from .. import examples
import sys,os,time,subprocess,exconfig
from numpy import arange

if os.environ['QUTIP_GUI']=="PYSIDE":
    from PySide import QtGui, QtCore

elif os.environ['QUTIP_GUI']=="PYQT4":
    from PyQt4 import QtGui, QtCore

class Examples(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        #WINDOW PROPERTIES
        self.setWindowTitle('QuTiP Examples')
        self.resize(790, 650)
        self.center()
        self.setFocus()
        #self.setAttribute(QtCore.Qt.WA_TranslucentBackground)#transparent
        #self.setWindowOpacity(.9)
        #self.setWindowFlags(Qt.Popup)#no titlebar
        
        #IMAGES--------------------
        
        #TEXT--------------------
        tlabel = QtGui.QLabel(self)
        tlabel.setStyleSheet("QLabel {font-weight: bold;font-size: 20px;}")
        tlabel.setText("QuTiP Example Scripts:")
        tlabel.move(280, 20)
        #
        utext = QtGui.QLabel(self)
        utext.setStyleSheet("QLabel {font-weight: bold;font-size: 14px;}")
        utext.setText("Click on the link to view the webpage associated with each script.")
        utext.move(150, 50)
        #
        alabel = QtGui.QLabel(self)
        alabel.setStyleSheet("QLabel {font-weight: bold;font-size: 10px;}")
        alabel.setText("Copyright (c) 2011, Paul D. Nation & Robert J. Johansson")
        alabel.move(5, 630)
        #-----EXAMPLE BUTTONS-----------------
        
        #ROW 1 ################################
        y=100
        #COLUMN 1
        x=20
        testqobj = QtGui.QPushButton('Test Qobj\n Algebra', self)
        testqobj .setGeometry(x, y, 150, 80)
        testqobj .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(testqobj, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('hello()'))
        testqobjlabel = QtGui.QLabel(self)
        testqobjlabel.setOpenExternalLinks(True)
        testqobjlabel.setText("<a href=http://code.google.com/p/qutip>TestQobj</a>")
        testqobjlabel.move(65, 190)
        #COLUMN 2
        x=170
        wigcat = QtGui.QPushButton('Wigner function:\nSchr. cat state', self)
        wigcat.setGeometry(x, y, 150, 80)
        wigcat.setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(wigcat, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('wignercat()'))
        wigcatlabel = QtGui.QLabel(self)
        wigcatlabel.setOpenExternalLinks(True)
        wigcatlabel.setText("<a href=http://code.google.com/p/qutip/wiki/ExamplesSchCatDist>SchCatDist</a>")
        wigcatlabel.move(215, 190)
        #COLUMN 3
        x=320
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri.setGeometry(x, y, 150, 80)
        montetri.setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('montetri()'))
        #column 4
        x=470
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri.setGeometry(x, y, 150, 80)
        montetri.setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('montetri()'))
        #column 5
        x=620
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri.setGeometry(x, y, 150, 80)
        montetri.setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('montetri()'))
        ###################################
        
        #ROW 2 ############################
        y=220
        #COLUMN 1
        x=20
        testqobj = QtGui.QPushButton('Test Qobj\n Algebra', self)
        testqobj.setGeometry(x,y, 150, 80)
        testqobj.setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(testqobj, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('hello()'))
        #COLUMN 2
        x=170
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri .setGeometry(x,y, 150, 80)
        montetri .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('montetri()'))
        #COLUMN 3
        x=320
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri .setGeometry(x,y, 150, 80)
        montetri .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('montetri()'))
        #column 4
        x=470
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri .setGeometry(x,y, 150, 80)
        montetri .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('montetri()'))
        #column 5
        x=620
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri .setGeometry(x,y, 150, 80)
        montetri .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('montetri()'))
        ######################################
        
        
        #ROW 3 ###############################
        y=340
        #COLUMN 1
        x=20
        testqobj = QtGui.QPushButton('Test Qobj\n Algebra', self)
        testqobj .setGeometry(x,y, 150, 80)
        testqobj .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(testqobj, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('hello()'))
        #COLUMN 2
        x=170
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri .setGeometry(x,y, 150, 80)
        montetri .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('montetri()'))
        #COLUMN 3
        x=320
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri .setGeometry(x,y, 150, 80)
        montetri .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('montetri()'))
        #column 4
        x=470
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri .setGeometry(x,y, 150, 80)
        montetri .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('montetri()'))
        #column 5
        x=620
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri .setGeometry(x,y, 150, 80)
        montetri .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('montetri()'))
        ####################################
        
        
        #ROW 4 #########################
        y=460
        #COLUMN 1
        x=20
        testqobj = QtGui.QPushButton('Test Qobj\n Algebra', self)
        testqobj .setGeometry(x,y, 150, 80)
        testqobj .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(testqobj, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('hello()'))
        #COLUMN 2
        x=170
        montetri = QtGui.QPushButton('Monte-Carlo:\n trilinear Hamilt.', self)
        montetri .setGeometry(x,y, 150, 80)
        montetri .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('trilinearmc()'))
        #COLUMN 3
        x=320
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri .setGeometry(x,y, 150, 80)
        montetri .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('montetri()'))
        #column 4
        x=470
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri .setGeometry(x,y, 150, 80)
        montetri .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('montetri()'))
        #column 5
        x=620
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri .setGeometry(x,y, 150, 80)
        montetri .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('montetri()'))
        ###############################
        
        #QUIT BUTTON-----------------
        quit = QtGui.QPushButton('Close', self)
        quit.setGeometry(670, 570, 100, 60)
        quit.setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(quit, QtCore.SIGNAL('clicked()'),QtGui.qApp, QtCore.SLOT('quit()'))
    
    def center(self):
        screen = QtGui.QDesktopWidget().screenGeometry()
        size =  self.geometry()
        self.move((screen.width()-size.width())/2, (screen.height()-size.height())/2)
    def moveout(self):
        self.clearFocus()
        for op in arange(0.9,-0.1,-0.1):
            time.sleep(.02)
            self.setWindowOpacity(op)
        self.setVisible(False)
    def movein(self):
        self.setFocus()
        self.setVisible(True)
        for op in arange(0.1,1.1,0.1):
            time.sleep(.02)
            self.setWindowOpacity(op)
    def trilinearmc(self):
        self.moveout()
        exconfig.option=5
        self.close()
    def hello(self):
        self.moveout()
        print 'monte'
        time.sleep(1)
        self.movein()
    def wignercat(self):
        self.moveout()
        exconfig.option=2
        self.close()



    
    
