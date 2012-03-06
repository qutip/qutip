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
import os,sys,threading
from scipy import array,ceil,remainder
from multiprocessing import Pool,cpu_count
import datetime

if os.environ['QUTIP_GRAPHICS']=="NO":
    raise RuntimeError('No graphics installed or available.')
        
if sys.platform.startswith("darwin"):#needed for PyQt on mac (because of pyobc) 
    import Foundation

if os.environ['QUTIP_GUI']=="PYSIDE":
    from PySide import QtCore,QtGui
    Signal=QtCore.Signal
elif os.environ['QUTIP_GUI']=="PYQT4":
    from PyQt4 import QtCore,QtGui
    Signal=QtCore.pyqtSignal

class Sig(QtCore.QObject):
    finish = Signal()
    trajdone = Signal()

class ProgressBar(QtGui.QWidget):
    def __init__(self,top,thread,mx,ncpus,parent = None):
        QtGui.QWidget.__init__(self, parent)
        self.setWindowFlags(QtCore.Qt.Window|QtCore.Qt.CustomizeWindowHint|QtCore.Qt.WindowTitleHint|QtCore.Qt.WindowMinimizeButtonHint)
        self.wait=ncpus
        self.top=top
        self.max=mx
        self.st=datetime.datetime.now()
        self.num = 0
        self.pbar = QtGui.QProgressBar(self)
        self.pbar.setStyleSheet("QProgressBar {width: 25px;border: 3px solid black; border-radius: 5px; background: white;text-align: center;padding: 0px;}" 
                               +"QProgressBar::chunk:horizontal {background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #00CCEE, stop: 0.3 #00DDEE, stop: 0.6 #00EEEE, stop:1 #00FFEE);}")
        self.pbar.setGeometry(25, 40, 300,40)
        self.label = QtGui.QLabel(self)
        self.label.setStyleSheet("QLabel {font-size: 12px;}")
        self.label.setText("Trajectories completed:                                       ")
        self.label.move(25, 20)
        self.estlabel = QtGui.QLabel(self)
        self.estlabel.setStyleSheet("QLabel {font-size: 12px;}")
        self.estlabel.setText("                                                           ")
        self.estlabel.move(25, 82)
        self.setWindowTitle('Monte-Carlo Trajectories on '+str(self.wait)+" CPUs")
        self.setGeometry(300, 300, 350, 120)
        #self.setWindowOpacity(0.9) #make transparent
        screen = QtGui.QDesktopWidget().screenGeometry()
        size =  self.geometry()
        self.move((screen.width()-size.width())/2, (screen.height()-size.height())/2)
        self.setFixedSize(self.size());   
        self.thread=thread
        self.sig=Sig()
        self.sig.trajdone.connect(self.updates)
        self.sig.finish.connect(self.end)
    def updates(self):
        self.num+=1
        self.pbar.setValue((100.0*self.num)/self.max)
        self.label.setText("Trajectories completed: "+ str(self.num)+"/"+str(self.max))
        if self.num>=self.wait and remainder(self.num,self.wait)==0:
            nwt=datetime.datetime.now()
            diff=((nwt.day-self.st.day)*86400+(nwt.hour-self.st.hour)*(60**2)+(nwt.minute-self.st.minute)*60+(nwt.second-self.st.second))*(self.max-self.num)/(1.0*self.num)
            secs=datetime.timedelta(seconds=ceil(diff))
            dd = datetime.datetime(1,1,1) + secs
            time_string="%02d:%02d:%02d:%02d" % (dd.day-1,dd.hour,dd.minute,dd.second)
            self.estlabel.setText("Est. time remaining: "+time_string)
    def run(self):
        self.thread.start()
    def end(self):
        return self.close()





class Pthread(QtCore.QThread):
    def __init__(self,target=None,args=None,top=None,parent=None):
        QtCore.QThread.__init__(self,parent)
        self.target = target
        self.args=args
        self.top=top
        self.exiting = True
    def run(self):
        if sys.platform.startswith("darwin"):#needed for PyQt on mac (because of pyobc) 
            pool = Foundation.NSAutoreleasePool.alloc().init()
        self.target(self.args,self)
        return self.top.bar.sig.finish.emit()
    def callback(self,args):
        self.top.callback(args)
        self.top.bar.sig.trajdone.emit()





 