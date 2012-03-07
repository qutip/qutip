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
import qutip.examples
import sys,os,time
from numpy import arange
from qutip.examples import exconfig

if os.environ['QUTIP_GUI']=="PYSIDE":
    from PySide import QtGui, QtCore

elif os.environ['QUTIP_GUI']=="PYQT4":
    from PyQt4 import QtGui, QtCore

class Examples(QtGui.QWidget):
    def __init__(self,version,parent=None):
        QtGui.QWidget.__init__(self, parent)
        #WINDOW PROPERTIES
        self.setWindowTitle('QuTiP Examples')
        self.resize(790, 650)
        self.setMinimumSize(790, 650)
        self.setMaximumSize(790, 650)
        self.center()
        self.setFocus()
        #self.setWindowFlags(QtCore.Qt.Popup)#no titlebar 
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
        u2text.setText("A growing list of examples may be found at the QuTiP homepage: "+"<a href=http://qutip.googlecode.com/svn/doc/"+version+"/html/examples/examples.html>QuTip Examples</a>")
        u2text.move(130, 65)
        #
        alabel = QtGui.QLabel(self)
        alabel.setStyleSheet("QLabel {font-weight: bold;font-size: 10px;}")
        alabel.setText("Copyright (c) 2011-2012, Paul D. Nation & Robert J. Johansson")
        alabel.move(5, 630)
        
        #QUIT BUTTON-----------------
        quit = QtGui.QPushButton('Close', self)
        quit.setGeometry(670, 578, 100, 60)
        quit.setFocusPolicy(QtCore.Qt.NoFocus)
        quit.clicked.connect(self.close)
        
        #-----EXAMPLE BUTTONS-----------------
        
        #ROW 1 ################################
        y=100
        #COLUMN 1
        x=20
        b11=QtGui.QPushButton('Qobj basics', self)
        b11.setGeometry(x, y, 150, 80)
        b11.setFocusPolicy(QtCore.Qt.NoFocus)
        b11.clicked.connect(self.button11)
        b11label = QtGui.QLabel(self)
        b11label.setOpenExternalLinks(True)
        b11label.setText("<a href=http://qutip.googlecode.com/svn/doc/"+version+"/html/guide/guide-basics.html>Basics Guide</a>")
        b11label.move(58, 185)
        #COLUMN 2
        x=170
        b12 = QtGui.QPushButton('Manipulating\nstates/operators', self)
        b12.setGeometry(x, y, 150, 80)
        b12.setFocusPolicy(QtCore.Qt.NoFocus)
        b12.clicked.connect(self.button12)
        b12label = QtGui.QLabel(self)
        b12label.setOpenExternalLinks(True)
        b12label.setText("<a href=http://qutip.googlecode.com/svn/doc/"+version+"/html/guide/guide-states.html>Operators & States</a>")
        b12label.move(187, 185)
        #COLUMN 3
        x=320
        b13 = QtGui.QPushButton('tensor and ptrace\nfunctions', self)
        b13.setGeometry(x, y, 150, 80)
        b13.setFocusPolicy(QtCore.Qt.NoFocus)
        b13.clicked.connect(self.button13)
        b13label = QtGui.QLabel(self)
        b13label.setOpenExternalLinks(True)
        b13label.setText("<a href=http://qutip.googlecode.com/svn/doc/"+version+"/html/guide/guide-tensor.html>Composite States</a>")
        b13label.move(345, 185)
        #column 4
        x=470
        b14 = QtGui.QPushButton('Schrodinger cat\n Wigner and Q-func.', self)
        b14.setGeometry(x, y, 150, 80)
        b14.setFocusPolicy(QtCore.Qt.NoFocus)
        b14.clicked.connect(self.button14)
        b14label = QtGui.QLabel(self)
        b14label.setOpenExternalLinks(True)
        b14label.setText("<a href=http://qutip.googlecode.com/svn/doc/"+version+"/html/examples/examples-schcatdist.html>Schrodinger's Cat</a>")
        b14label.move(492, 185)
        #column 5
        x=620
        b15 = QtGui.QPushButton('Constructing a\nsqueezed state', self)
        b15.setGeometry(x, y, 150, 80)
        b15.setFocusPolicy(QtCore.Qt.NoFocus)
        b15.clicked.connect(self.button15)
        b15label = QtGui.QLabel(self)
        b15label.setOpenExternalLinks(True)
        b15label.setText("<a href=http://qutip.googlecode.com/svn/doc/"+version+"/html/examples/examples-squeezed.html>Squeezed State</a>")
        b15label.move(647, 185)
        ###################################
        
        #ROW 2 ############################
        y=220
        #COLUMN 1
        x=20
        b21 = QtGui.QPushButton('Steady state:\ncavity+qubit', self)
        b21.setGeometry(x,y, 150, 80)
        b21.setFocusPolicy(QtCore.Qt.NoFocus)
        b21.clicked.connect(self.button21)
        b21label = QtGui.QLabel(self)
        b21label.setOpenExternalLinks(True)
        b21label.setText("<a href=http://qutip.googlecode.com/svn/doc/"+version+"/html/examples/examples-drivencavitysteady.html>Driven Cavity</a>")
        b21label.move(50, 305)
        #COLUMN 2
        x=170
        b22 = QtGui.QPushButton('Steady state:\nthermal envir.', self)
        b22 .setGeometry(x,y, 150, 80)
        b22 .setFocusPolicy(QtCore.Qt.NoFocus)
        b22.clicked.connect(self.button22)
        b22label = QtGui.QLabel(self)
        b22label.setOpenExternalLinks(True)
        b22label.setText("<a href=http://qutip.googlecode.com/svn/doc/"+version+"/html/guide/guide-steady.html>Thermal Environment</a>")
        b22label.move(179, 305)
        #COLUMN 3
        x=320
        b23 = QtGui.QPushButton('Eseries', self)
        b23 .setGeometry(x,y, 150, 80)
        b23 .setFocusPolicy(QtCore.Qt.NoFocus)
        b23.clicked.connect(self.button23)
        b23label = QtGui.QLabel(self)
        b23label.setOpenExternalLinks(True)
        b23label.setText("<a href=http://qutip.googlecode.com/svn/doc/"+version+"/html/guide/guide-eseries.html>Eseries</a>")
        b23label.move(x+55, 305)
        #column 4
        x=470
        b24 = QtGui.QPushButton('Master equation:\nRabi oscillations', self)
        b24.setGeometry(x,y, 150, 80)
        b24.setFocusPolicy(QtCore.Qt.NoFocus)
        b24.clicked.connect(self.button24)
        b24label = QtGui.QLabel(self)
        b24label.setOpenExternalLinks(True)
        b24label.setText("<a href=http://qutip.googlecode.com/svn/doc/"+version+"/html/examples/examples-jc-model.html>Vacuum Rabi</a>")
        b24label.move(505, 305)
        #column 5
        x=620
        b25 = QtGui.QPushButton('Master equation:\nSingle-atom laser', self)
        b25.setGeometry(x,y, 150, 80)
        b25.setFocusPolicy(QtCore.Qt.NoFocus)
        b25.clicked.connect(self.button25)
        b25label = QtGui.QLabel(self)
        b25label.setOpenExternalLinks(True)
        b25label.setText("<a href=http://qutip.googlecode.com/svn/doc/"+version+"/html/examples/examples-lasing.html>Single-Atom Laser</a>")
        b25label.move(635, 305)
        ######################################
        
        
        #ROW 3 ###############################
        y=340
        #COLUMN 1
        x=20
        b31 = QtGui.QPushButton('Density matrix\n metrics: Fidelity', self)
        b31 .setGeometry(x,y, 150, 80)
        b31 .setFocusPolicy(QtCore.Qt.NoFocus)
        b31.clicked.connect(self.button31)
        b31label = QtGui.QLabel(self)
        b31label.setOpenExternalLinks(True)
        b31label.setText("<a href=http://qutip.googlecode.com/svn/doc/"+version+"/html/examples/examples-fidelity.html>Fidelity</a>")
        b31label.move(72, 425)
        #COLUMN 2
        x=170
        b32 = QtGui.QPushButton('Propagator:\nSteady state of\na driven system', self)
        b32 .setGeometry(x,y, 150, 80)
        b32 .setFocusPolicy(QtCore.Qt.NoFocus)
        b32.clicked.connect(self.button32)
        b32label = QtGui.QLabel(self)
        b32label.setOpenExternalLinks(True)
        b32label.setText("<a href=http://qutip.googlecode.com/svn/doc/"+version+"/html/examples/examples-propagator.html>Propagator</a>")
        b32label.move(x+40, 425)
        #COLUMN 3
        x=320
        b33 = QtGui.QPushButton('Heisenberg\n spin chain (N=4)', self)
        b33 .setGeometry(x,y, 150, 80)
        b33 .setFocusPolicy(QtCore.Qt.NoFocus)
        b33.clicked.connect(self.button33)
        b33label = QtGui.QLabel(self)
        b33label.setOpenExternalLinks(True)
        b33label.setText("<a href=http://qutip.googlecode.com/svn/doc/"+version+"/html/examples/examples-spinchain.html>Spin Chain</a>")
        b33label.move(365, 425)
        #column 4
        x=470
        b34 = QtGui.QPushButton('Correlations\n and spectrum', self)
        b34 .setGeometry(x,y, 150, 80)
        b34 .setFocusPolicy(QtCore.Qt.NoFocus)
        b34.clicked.connect(self.button34)
        b34label = QtGui.QLabel(self)
        b34label.setOpenExternalLinks(True)
        b34label.setText("<a href=http://qutip.googlecode.com/svn/doc/"+version+"/html/examples/examples-corrfunc.html>Correl.</a> / <a href=http://qutip.googlecode.com/svn/doc/"+version+"/html/examples/examples-spectrumsteady.html>Spectrum</a>")
        b34label.move(488, 425)
        #column 5
        x=620
        b35 = QtGui.QPushButton('Qubit decay on\nBloch sphere', self)
        b35 .setGeometry(x,y, 150, 80)
        b35 .setFocusPolicy(QtCore.Qt.NoFocus)
        b35.clicked.connect(self.button35)
        b34label = QtGui.QLabel(self)
        b34label.setOpenExternalLinks(True)
        b34label.setText("<a href=http://qutip.googlecode.com/svn/doc/"+version+"/html/examples/examples-bloch-qubit-decay.html>Bloch Decay</a>")
        b34label.move(660, 425)
        ####################################
        
        
        #ROW 4 #########################
        y=460
        #COLUMN 1
        x=20
        b41 = QtGui.QPushButton('Monte-Carlo:\ncavity+qubit', self)
        b41 .setGeometry(x,y, 150, 80)
        b41 .setFocusPolicy(QtCore.Qt.NoFocus)
        b41.clicked.connect(self.button41)
        b41label = QtGui.QLabel(self)
        b41label.setOpenExternalLinks(True)
        b41label.setText("<a href=http://qutip.googlecode.com/svn/doc/"+version+"/html/examples/examples-expectmonte.html>MC Cavity+Qubit</a>")
        b41label.move(40, 545)
        #COLUMN 2
        x=170
        b42 = QtGui.QPushButton('Monte-Carlo:\ntrilinear Hamilt.', self)
        b42 .setGeometry(x,y, 150, 80)
        b42 .setFocusPolicy(QtCore.Qt.NoFocus)
        b42.clicked.connect(self.button42)
        b42label = QtGui.QLabel(self)
        b42label.setOpenExternalLinks(True)
        b42label.setText("<a href=http://qutip.googlecode.com/svn/doc/"+version+"/html/examples/examples-trilinearmonte.html>Trilinear Monte-Carlo</a>")
        b42label.move(173, 545)
        #COLUMN 3
        x=320
        b43 = QtGui.QPushButton('Monte-Carlo:\nthermal deviations', self)
        b43 .setGeometry(x,y, 150, 80)
        b43 .setFocusPolicy(QtCore.Qt.NoFocus)
        b43.clicked.connect(self.button43)
        b43label = QtGui.QLabel(self)
        b43label.setOpenExternalLinks(True)
        b43label.setText("<a href=http://qutip.googlecode.com/svn/doc/"+version+"/html/examples/examples-thermalmonte.html>Trilinear Non-thermal</a>")
        b43label.move(325, 545)
        #column 4
        x=470
        b44 = QtGui.QPushButton('Time-dependent\nHamiltonians:\nRabi oscillations', self)
        b44 .setGeometry(x,y, 150, 80)
        b44 .setFocusPolicy(QtCore.Qt.NoFocus)
        b44.clicked.connect(self.button44)
        b44label = QtGui.QLabel(self)
        b44label.setOpenExternalLinks(True)
        b44label.setText("<a href=http://qutip.googlecode.com/svn/doc/"+version+"/html/examples/examples-time-dependence.html>Rabi Oscillations</a>")
        b44label.move(x+27, 545)
        #column 5
        x=620
        b45 = QtGui.QPushButton('Time-dependent\nHamiltonians:\nLZ transitions', self)
        b45 .setGeometry(x,y, 150, 80)
        b45 .setFocusPolicy(QtCore.Qt.NoFocus)
        b45.clicked.connect(self.button45)
        b45label = QtGui.QLabel(self)
        b45label.setOpenExternalLinks(True)
        b45label.setText("<a href=http://qutip.googlecode.com/svn/doc/"+version+"/html/examples/examples-landau-zener.html>Landau-Zener</a>")
        b45label.move(x+35, 545)
        ###############################
    def center(self):
        screen = QtGui.QDesktopWidget().screenGeometry()
        size =  self.frameSize()
        self.move((screen.width()-size.width())/2, (screen.height()-size.height())/2)
    def moveout(self):
        #self.clearFocus()
        for op in arange(0.9,-0.1,-0.1):
            time.sleep(.02)
            self.setWindowOpacity(op)
        self.setVisible(False)
    def movein(self):
        self.focusWidget(True)
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


class HoverButton(QtGui.QPushButton): 
    def enterEvent(self,event):  
        self.setStyleSheet('QPushButton {border-width: 2px;border-color:#222222;border-style: solid;border-radius: 7}')
    def leaveEvent(self,event):  
        self.setStyleSheet('QPushButton {border-width: 2px;border-color:#888888;border-style: solid;border-radius: 7}')  
