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
import sys,os,time
from numpy import arange
from ..examples import exconfig
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
        tlabel.move(280, 10)
        #
        utext = QtGui.QLabel(self)
        utext.setStyleSheet("QLabel {font-weight: bold;font-size: 14px;}")
        utext.setText("Click on the link to view the webpage associated with each script.")
        utext.move(150, 40)
        #
        u2text = QtGui.QLabel(self)
        u2text.setStyleSheet("QLabel {font-weight: bold;font-size: 12px;}")
        u2text.setOpenExternalLinks(True)
        u2text.setText("A growing list of examples may be found at the QuTip homepage: "+"<a href=http://code.google.com/p/qutip/wiki/Examples>QuTip Examples</a>")
        u2text.move(130, 65)
        #
        alabel = QtGui.QLabel(self)
        alabel.setStyleSheet("QLabel {font-weight: bold;font-size: 10px;}")
        alabel.setText("Copyright (c) 2011, Paul D. Nation & Robert J. Johansson")
        alabel.move(5, 630)
        
        #QUIT BUTTON-----------------
        quit = QtGui.QPushButton('Close', self)
        quit.setGeometry(670, 570, 100, 60)
        quit.setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(quit, QtCore.SIGNAL('clicked()'),QtGui.qApp, QtCore.SLOT('quit()'))
        
        #-----EXAMPLE BUTTONS-----------------
        
        #ROW 1 ################################
        y=100
        #COLUMN 1
        x=20
        testqobj = QtGui.QPushButton('Test Qobj\n operations', self)
        testqobj .setGeometry(x, y, 150, 80)
        testqobj .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(testqobj, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('button11()'))
        testqobjlabel = QtGui.QLabel(self)
        testqobjlabel.setOpenExternalLinks(True)
        testqobjlabel.setText("<a href=http://code.google.com/p/qutip>TestQobj</a>")
        testqobjlabel.move(65, 190)
        #COLUMN 2
        x=170
        wigcat = QtGui.QPushButton('Wigner function:\nSchr. cat state', self)
        wigcat.setGeometry(x, y, 150, 80)
        wigcat.setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(wigcat, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('button12()'))
        wigcatlabel = QtGui.QLabel(self)
        wigcatlabel.setOpenExternalLinks(True)
        wigcatlabel.setText("<a href=http://code.google.com/p/qutip/wiki/ExamplesSchCatDist>SchCatDist</a>")
        wigcatlabel.move(215, 190)
        #COLUMN 3
        x=320
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri.setGeometry(x, y, 150, 80)
        montetri.setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('button13()'))
        #column 4
        x=470
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri.setGeometry(x, y, 150, 80)
        montetri.setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('button14()'))
        #column 5
        x=620
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri.setGeometry(x, y, 150, 80)
        montetri.setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('button15()'))
        ###################################
        
        #ROW 2 ############################
        y=220
        #COLUMN 1
        x=20
        testqobj = QtGui.QPushButton('Test Qobj\n Algebra', self)
        testqobj.setGeometry(x,y, 150, 80)
        testqobj.setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(testqobj, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('button21()'))
        #COLUMN 2
        x=170
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri .setGeometry(x,y, 150, 80)
        montetri .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('button22()'))
        #COLUMN 3
        x=320
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri .setGeometry(x,y, 150, 80)
        montetri .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('button23()'))
        #column 4
        x=470
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri .setGeometry(x,y, 150, 80)
        montetri .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('button24()'))
        #column 5
        x=620
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri .setGeometry(x,y, 150, 80)
        montetri .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('button25()'))
        ######################################
        
        
        #ROW 3 ###############################
        y=340
        #COLUMN 1
        x=20
        testqobj = QtGui.QPushButton('Test Qobj\n Algebra', self)
        testqobj .setGeometry(x,y, 150, 80)
        testqobj .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(testqobj, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('button31()'))
        #COLUMN 2
        x=170
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri .setGeometry(x,y, 150, 80)
        montetri .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('button32()'))
        #COLUMN 3
        x=320
        montetri = QtGui.QPushButton('Heisenberg\n spin chain (N=4)', self)
        montetri .setGeometry(x,y, 150, 80)
        montetri .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('button33()'))
        #column 4
        x=470
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri .setGeometry(x,y, 150, 80)
        montetri .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('button34()'))
        #column 5
        x=620
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri .setGeometry(x,y, 150, 80)
        montetri .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('button35()'))
        ####################################
        
        
        #ROW 4 #########################
        y=460
        #COLUMN 1
        x=20
        testqobj = QtGui.QPushButton('Test Qobj\n Algebra', self)
        testqobj .setGeometry(x,y, 150, 80)
        testqobj .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(testqobj, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('button41()'))
        #COLUMN 2
        x=170
        montetri = QtGui.QPushButton('Monte-Carlo:\n trilinear Hamilt.', self)
        montetri .setGeometry(x,y, 150, 80)
        montetri .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('button42()'))
        #COLUMN 3
        x=320
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri .setGeometry(x,y, 150, 80)
        montetri .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('button43()'))
        #column 4
        x=470
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri .setGeometry(x,y, 150, 80)
        montetri .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('button44()'))
        #column 5
        x=620
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri .setGeometry(x,y, 150, 80)
        montetri .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('button45()'))
        ###############################
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
    #first row button pressed
    def button11(self):
        self.moveout()
        exconfig.option=11
        self.close()
    def button12(self):
        self.moveout()
        exconfig.option=12
        self.close()
    def button13(self):
        self.moveout()
        exconfig.option=13
        self.close()
    def button14(self):
        self.moveout()
        exconfig.option=14
        self.close()
    def button15(self):
        self.moveout()
        exconfig.option=15
        self.close()
    #second row button pressed
    def button21(self):
        self.moveout()
        exconfig.option=21
        self.close()
    def button22(self):
        self.moveout()
        exconfig.option=22
        self.close()
    def button23(self):
        self.moveout()
        exconfig.option=23
        self.close()
    def button24(self):
        self.moveout()
        exconfig.option=24
        self.close()
    def button25(self):
        self.moveout()
        exconfig.option=25
        self.close()
    #third row button pressed
    def button31(self):
        self.moveout()
        exconfig.option=31
        self.close()
    def button32(self):
        self.moveout()
        exconfig.option=32
        self.close()
    def button33(self):
        self.moveout()
        exconfig.option=33
        self.close()
    def button34(self):
        self.moveout()
        exconfig.option=34
        self.close()
    def button35(self):
        self.moveout()
        exconfig.option=35
        self.close()
    #forth row button pressed
    def button41(self):
        self.moveout()
        exconfig.option=41
        self.close()
    def button42(self):
        self.moveout()
        exconfig.option=42
        self.close()
    def button43(self):
        self.moveout()
        exconfig.option=43
        self.close()
    def button44(self):
        self.moveout()
        exconfig.option=44
        self.close()
    def button45(self):
        self.moveout()
        exconfig.option=45
        self.close()
    



    
    
