import sys,time,threading
from multiprocessing import Pool

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
        self.time=time.time()
        self.num = 0
        self.pbar = QtGui.QProgressBar(self)
        self.pbar.setStyleSheet("QProgressBar {width: 25px;border: 3px solid black; border-radius: 5px; background: white;text-align: center;padding: 0px;}" 
                               +"QProgressBar::chunk:horizontal {background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #6699FF, stop: 0.5 #66AAFF, stop: 0.6 #66CCFF, stop:1 #FFFFFF);}")
        self.pbar.setGeometry(25, 40, 300,40)
        self.label = QtGui.QLabel(self)
        self.label.setStyleSheet("QLabel {font-size: 12px;}")
        self.label.setText("Trajectories completed:                                       ")
        self.label.move(25, 20)
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
    def run(self):
        self.thread.start()
    def end(self):
        return self.close()





class Thread(QtCore.QThread):
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





 