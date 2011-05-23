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
import sys,time,threading
from multiprocessing import Pool

try:
    from PySide.QtCore import *
    from PySide.QtGui import *
except:
    try:
        from PyQt4.QtCore import *
        from PyQt4.QtGui import *
    except:
        raise RuntimeError('PyQt4 or PySide GUI module is not installed.')

if sys.platform.startswith("darwin"):#needed for PyQt on mac (because of pyobc) 
    import Foundation

class ProgressBar(QWidget):

    def __init__(self, parent = None):
        QWidget.__init__(self, parent)
        self.setWindowFlags(Qt.Window|Qt.CustomizeWindowHint|Qt.WindowTitleHint|Qt.WindowMinimizeButtonHint)
        self.num = 0
        self.pbar = QProgressBar(self)
        self.pbar.setStyleSheet("QProgressBar {width: 25px;border: 3px solid black; border-radius: 5px; background: white;text-align: center;padding: 0px;}" 
                               +"QProgressBar::chunk:horizontal {background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #6699FF, stop: 0.5 #66AAFF, stop: 0.6 #66CCFF, stop:1 #FFFFFF);}")
        self.pbar.setGeometry(25, 40, 300,40)
        self.label = QLabel(self)
        self.label.setStyleSheet("QLabel {font-size: 12px;}")
        self.label.setText("Trajectories completed:                                       ")
        self.label.move(25, 20)
        self.setWindowTitle('Monte-Carlo Trajectories')
        self.setGeometry(300, 300, 350, 120)
        #self.setWindowOpacity(0.9) #make transparent
        screen = QDesktopWidget().screenGeometry()
        size =  self.geometry()
        self.move((screen.width()-size.width())/2, (screen.height()-size.height())/2)
        self.setFixedSize(self.size());
        
        self.thread=Thread(top=self,target=func,parent=None)
        self.connect(self.thread, SIGNAL("completed"), self.update)
        self.connect(self.thread, SIGNAL("done"), self.end)
    def update(self,*args):
        # Old style: emits the signal using the SIGNAL function.
        self.pbar.setValue(self.num)
        self.label.setText("Trajectories completed: "+ str(self.num)+"/100")
        self.num+=1
    def run(self):
        self.thread.start()
    def end(self):
        time.sleep(1)
        return self.close()

class Thread(QThread):
    def __init__(self, top=None,target=None,parent=None):
        QThread.__init__(self,parent)
        self.target = target
    def run(self):
        if sys.platform.startswith("darwin"):#needed for PyQt on mac (because of pyobc) 
            pool = Foundation.NSAutoreleasePool.alloc().init()
        self.target(self)
        return
    def done(self,*args):
        self.emit(SIGNAL("completed"))

def func(top):
    from multiprocessing import Pool
    p=Pool(processes=2)
    for nt in xrange(101):
        p.apply_async(f,args=(nt,),callback=top.done)
    p.close()
    p.join()
    top.emit(SIGNAL("done"))


def f(x):
    #import time,scipy
    time.sleep(.05)
    #print time.time()
    return 


    
    


app = QApplication(sys.argv)
abox = ProgressBar()
QTimer.singleShot(0,abox.run)
abox.show()
abox.raise_()
app.exec_()



 