import sys,os,time
try:
    from PySide import QtGui, QtCore
    from PySide.QtCore import *
except:
    try:
        from PyQt4 import QtGui, QtCore
        from PyQt4.QtCore import *
    except:
        raise TypeError('no graphics installed')
    
import numpy,scipy,matplotlib

CD_BASE = os.path.dirname(__file__)
class AboutBox(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        #WINDOW PROPERTIES
        self.setWindowTitle('About QuTiP')
        self.resize(400, 450)
        self.center()
        self.setFocus()
        #self.setAttribute(Qt.WA_TranslucentBackground)#transparent
        #self.setWindowOpacity(0.95)
        #self.setWindowFlags(Qt.Popup)#no titlebar
        #self.pic=QtGui.QLabel(self)
        #self.pic.setGeometry(0, 0, 450, 500)
        #self.pic.setPixmap(QtGui.QPixmap(CD_BASE + "/about.png"))
        #IMAGES--------------------
        logo=QtGui.QLabel(self)
        logo.setGeometry(100, 0, 200, 163)
        logo.setPixmap(QtGui.QPixmap(CD_BASE + "/logo.png"))
        #TEXT--------------------
        tlabel = QtGui.QLabel(self)
        tlabel.setStyleSheet("QLabel {font-weight: bold;font-size: 18px;}")
        tlabel.setText("QuTiP: The Quantum Toolbox in Python")
        tlabel.move(15, 170)
        #
        label = QtGui.QLabel(self)
        label.setStyleSheet("QLabel {font-weight: bold;font-size: 12px;}")
        label.setText("QuTip Version:          "+str(matplotlib.__version__)+"\n"
                            +"NumPy Version:         "+str(numpy.__version__)+"\n"
                            +"SciPy Version:            "+str(scipy.__version__)+"\n"
                            +"MatPlotLib Version:   "+str(matplotlib.__version__))
        label.move(115, 210)
        #
        alabel = QtGui.QLabel(self)
        alabel.setStyleSheet("QLabel {font-weight: bold;font-size: 12px;}")
        alabel.setText("Copyright (c) 2011, Paul D. Nation & Robert J. Johansson")
        alabel.move(25, 300)
        #
        clabel = QtGui.QLabel(self)
        clabel.setStyleSheet("QLabel {font-weight: bold;font-size: 12px;}")
        clabel.setText("QuTiP is released under the GPL3.\n"
                        +"See the enclosed COPYING.txt\nfile for more information.")
        clabel.move(25, 330)
        #BUTTONS-----------------
        quit = QtGui.QPushButton('Close', self)
        quit.setGeometry(300, 390, 80, 40)
        quit.setFocusPolicy(QtCore.Qt.NoFocus)
        #quit.setStyleSheet("QPushButton {border: 1.5px solid black;border-radius: 10px;background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,stop: 0 #CCCCCC, stop: 1 #999999)}")
        self.connect(quit, QtCore.SIGNAL('clicked()'),QtGui.qApp, QtCore.SLOT('quit()'))
    def center(self):
        screen = QtGui.QDesktopWidget().screenGeometry()
        size =  self.geometry()
        self.move((screen.width()-size.width())/2, (screen.height()-size.height())/2)

app = QtGui.QApplication(sys.argv)
abox = AboutBox()
abox.show()
#QTimer.singleShot(2000,app.quit)
abox.raise_()
app.exec_()
print 'done'