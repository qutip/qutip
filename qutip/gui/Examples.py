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

import sys,os,time
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
        self.resize(790, 600)
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
        alabel = QtGui.QLabel(self)
        alabel.setStyleSheet("QLabel {font-weight: bold;font-size: 10px;}")
        alabel.setText("Copyright (c) 2011, Paul D. Nation & Robert J. Johansson")
        alabel.move(5, 580)
        #EXAMPLE BUTTONS-----------------
        
        #top row------------
        testqobj = QtGui.QPushButton('Test Qobj\n Algebra', self)
        testqobj .setGeometry(20, 100, 150, 80)
        testqobj .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(testqobj, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('hello()'))
        
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri .setGeometry(170, 100, 150, 80)
        montetri .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('montetri()'))
        
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri .setGeometry(320, 100, 150, 80)
        montetri .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('montetri()'))
        
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri .setGeometry(470, 100, 150, 80)
        montetri .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('montetri()'))
        
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri .setGeometry(620, 100, 150, 80)
        montetri .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('montetri()'))
        
        
        #second row------------
        testqobj = QtGui.QPushButton('Test Qobj\n Algebra', self)
        testqobj .setGeometry(20, 200, 150, 80)
        testqobj .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(testqobj, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('hello()'))
        
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri .setGeometry(170, 200, 150, 80)
        montetri .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('montetri()'))
        
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri .setGeometry(320, 200, 150, 80)
        montetri .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('montetri()'))
        
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri .setGeometry(470, 200, 150, 80)
        montetri .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('montetri()'))
        
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri .setGeometry(620, 200, 150, 80)
        montetri .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('montetri()'))
        
        
        #third row-------------
        testqobj = QtGui.QPushButton('Test Qobj\n Algebra', self)
        testqobj .setGeometry(20, 300, 150, 80)
        testqobj .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(testqobj, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('hello()'))
        
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri .setGeometry(170, 300, 150, 80)
        montetri .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('montetri()'))
        
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri .setGeometry(320, 300, 150, 80)
        montetri .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('montetri()'))
        
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri .setGeometry(470, 300, 150, 80)
        montetri .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('montetri()'))
        
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri .setGeometry(620, 300, 150, 80)
        montetri .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('montetri()'))
        
        
        #forth row-------
        testqobj = QtGui.QPushButton('Test Qobj\n Algebra', self)
        testqobj .setGeometry(20, 400, 150, 80)
        testqobj .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(testqobj, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('hello()'))
        
        montetri = QtGui.QPushButton('Monte-Carlo:\n trilinear Hamilt.', self)
        montetri .setGeometry(170, 400, 150, 80)
        montetri .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('montetri()'))
        
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri .setGeometry(320, 400, 150, 80)
        montetri .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('montetri()'))
        
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri .setGeometry(470, 400, 150, 80)
        montetri .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('montetri()'))
        
        montetri = QtGui.QPushButton('MC:\n trilinear', self)
        montetri .setGeometry(620, 400, 150, 80)
        montetri .setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(montetri, QtCore.SIGNAL('clicked()'),self, QtCore.SLOT('montetri()'))
        
        
        #QUIT BUTTON-----------------
        quit = QtGui.QPushButton('Close', self)
        quit.setGeometry(670, 520, 100, 60)
        quit.setFocusPolicy(QtCore.Qt.NoFocus)
        self.connect(quit, QtCore.SIGNAL('clicked()'),QtGui.qApp, QtCore.SLOT('quit()'))
    
    def center(self):
        screen = QtGui.QDesktopWidget().screenGeometry()
        size =  self.geometry()
        self.move((screen.width()-size.width())/2, (screen.height()-size.height())/2)
    def moveout(self):
        for op in arange(0.9,-0.1,-0.1):
            time.sleep(.02)
            self.setWindowOpacity(op)
        self.setVisible(False)
    def movein(self):
        self.setVisible(True)
        for op in arange(0.1,1.1,0.1):
            time.sleep(.02)
            self.setWindowOpacity(op)
    def montetri(self):
        self.moveout()
        print 'monte'
        time.sleep(1)
        self.movein()
    def hello(self):
        self.moveout()
        print 'hello'
        self.movein()



if __name__=="__main__":
    app = QtGui.QApplication(sys.argv)
    abox = Examples()
    abox.show()
    abox.raise_()
    app.exec_()
    
