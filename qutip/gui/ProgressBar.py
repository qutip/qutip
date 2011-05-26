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
from scipy import array,ceil
from multiprocessing import Pool
import datetime


try:
    from PySide import QtCore
    from PySide import QtGui
except:
    try:
        from PyQt4 import QtCore
        from PyQt4 import QtGui
    except:
        raise TypeError('no graphics installed')
        
if sys.platform.startswith("darwin"):#needed for PyQt on mac (because of pyobc) 
    import Foundation

class ProgressBar(QtGui.QWidget):
    def __init__(self,top,thread,mx,parent = None):
        QtGui.QWidget.__init__(self, parent)
        self.setWindowFlags(QtCore.Qt.Window|QtCore.Qt.CustomizeWindowHint|QtCore.Qt.WindowTitleHint|QtCore.Qt.WindowMinimizeButtonHint)
        self.top=top
        self.max=mx
        self.st=datetime.datetime.now()
        self.num = 0
        self.pbar = QtGui.QProgressBar(self)
        self.pbar.setStyleSheet("QProgressBar {width: 25px;border: 3px solid black; border-radius: 5px; background: white;text-align: center;padding: 0px;}" 
                               +"QProgressBar::chunk:horizontal {background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #6699FF, stop: 0.5 #66AAFF, stop: 0.6 #66CCFF, stop:1 #FFFFFF);}")
        self.pbar.setGeometry(25, 40, 300,40)
        self.label = QtGui.QLabel(self)
        self.label.setStyleSheet("QLabel {font-size: 12px;}")
        self.label.setText("Trajectories completed:                                       ")
        self.label.move(25, 20)
        self.estlabel = QtGui.QLabel(self)
        self.estlabel.setStyleSheet("QLabel {font-size: 12px;}")
        self.estlabel.setText("                                                           ")
        self.estlabel.move(25, 82)
        self.setWindowTitle('Monte-Carlo Trajectories')
        self.setGeometry(300, 300, 350, 120)
        #self.setWindowOpacity(0.9) #make transparent
        screen = QtGui.QDesktopWidget().screenGeometry()
        size =  self.geometry()
        self.move((screen.width()-size.width())/2, (screen.height()-size.height())/2)
        self.setFixedSize(self.size());   
        self.thread=thread
        self.connect(self.thread,QtCore.SIGNAL("completed"), self.update)
        self.connect(self.thread,QtCore.SIGNAL("done"), self.end)
    def update(self,*args):
        # Old style: emits the signal using the SIGNAL function.
        self.num+=1
        self.pbar.setValue((100.0*self.num)/self.max)
        self.label.setText("Trajectories completed: "+ str(self.num)+"/"+str(self.max))
        if self.num>10:
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
        return self.emit(QtCore.SIGNAL("done"))
    def callback(self,args):
        self.emit(QtCore.SIGNAL("completed"))
        self.top.callback(args)





 