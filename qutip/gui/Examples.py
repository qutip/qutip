import qutip.examples
import sys,os,time
from numpy import arange
from qutip.examples import exconfig

if os.environ['QUTIP_GUI']=="PYSIDE":
    from PySide import QtGui, QtCore

elif os.environ['QUTIP_GUI']=="PYQT4":
    from PyQt4 import QtGui, QtCore



class Examples(QtGui.QWidget):
    def center(self):
        screen = QtGui.QDesktopWidget().screenGeometry()
        size =  self.frameSize()
        self.move((screen.width()-size.width())/2, (screen.height()-size.height())/2)
    def moveout(self):
        #self.clearFocus()
        for op in arange(0.9,-0.1,-0.1):
            time.sleep(.02)
            self.setWindowOpacity(op)
        #self.setVisible(False)
    def movein(self):
        self.focusWidget()
        self.setVisible(True)
        for op in arange(0.1,1.1,0.1):
            time.sleep(.02)
            self.setWindowOpacity(op)
    
    def __init__(self,version,parent=None):
        QtGui.QWidget.__init__(self, parent)
        from demos_text import tab_labels,button_labels,button_desc,button_nums
        #WINDOW PROPERTIES
        self.setWindowTitle('QuTiP Examples')
        self.resize(790, 650)
        self.setMinimumSize(790, 650)
        self.setMaximumSize(790, 650)
        self.center()
        self.setFocus()
        mapper = QtCore.QSignalMapper(self)
        #self.setWindowFlags(QtCore.Qt.Popup)#no titlebar 
        #IMAGES--------------------
        
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setPointSize(22)
        fm = QtGui.QFontMetrics(font)
        
        
        #TEXT--------------------
        tlabel = QtGui.QLabel(self)
        t_string="QuTiP Example Scripts:"
        tlabel.setText(t_string)
        tlabel.setFont(font)
        pixelswide = fm.width(t_string)
        tlabel.move((self.width()-pixelswide)/2.0, 10)
        #
        utext = QtGui.QLabel(self)
        font.setPointSize(16)
        fm = QtGui.QFontMetrics(font)
        u_string="Click on the link to view the webpage associated with each script."
        utext.setText(u_string)
        utext.setFont(font)
        pixelswide = fm.width(u_string)
        utext.move((self.width()-pixelswide)/2.0, 40)
        #
        u2text = QtGui.QLabel(self)
        u2text.setOpenExternalLinks(True)
        font.setPointSize(14)
        fm = QtGui.QFontMetrics(font)
        u2_string="A growing list of examples may be found in the QuTiP documentation: "+"<a href=http://qutip.googlecode.com/svn/doc/"+version+"/html/examples/examples.html>QuTip Examples</a>"
        u2text.setText(u2_string)
        u2text.setFont(font)
        u2text.move((self.width()-pixelswide)/2.0-40, 65)
        #
        alabel = QtGui.QLabel(self)
        alabel.setStyleSheet("QLabel {font-family: Arial;font-weight: bold;font-size: 12px;}")
        alabel.setText("Copyright (c) 2011-2012, Paul D. Nation & Robert J. Johansson")
        alabel.move(5, 630)
        
        #QUIT BUTTON-----------------
        quit = HoverExit('Close', self)
        quit.setGeometry(700, 605, 80, 40)
        quit.setStyleSheet('QPushButton {font-family: Arial;border-width: 2px;border-color:#666666;border-style: solid;border-radius: 7;font-size: 16px;background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #FFAAAA, stop: 0.1 #FF9999, stop: 0.49 #FF8888, stop: 0.5 #FF7777, stop: 1 #FF6666)}')
        quit.setFocusPolicy(QtCore.Qt.NoFocus)
        quit.clicked.connect(self.close)
        
        #tab widget
        tab_widget = QtGui.QTabWidget(self) 
        tab_widget.move(10,100)
        tab_widget.resize(770,500)
        #tabs for tab widget
        num_tabs=len(tab_labels)
        tabs=[QtGui.QWidget(self) for k in range(num_tabs)]
        for k in range(num_tabs):
            tab_widget.addTab(tabs[k],tab_labels[k])
        
        #set tab button style
        tab_button_style='QPushButton {font-family: Arial;border-width: 2px;border-color:#666666;border-style: solid;border-radius: 7;background-color: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #00DDDD, stop: 0.1 #00CCDD, stop: 0.49 #00BBDD, stop: 0.5 #00AADD, stop: 1 #0099DD)}'
        
        
        #tab buttons
        tab_verts =[QtGui.QVBoxLayout(tabs[k]) for k in xrange(num_tabs)]
        num_elems=[len(button_labels[k]) for k in xrange(num_tabs)]
        tab_buttons=[[] for j in range(num_tabs)]
        for j in range(num_tabs):
            for k in range(num_elems[j]):
                button=HoverButton()
                button.setFont(font)
                button.setText(button_labels[j][k])
                button.setStyleSheet(tab_button_style)
                button.setFixedSize(335, 40)
                self.connect(button, QtCore.SIGNAL("clicked()"), mapper, QtCore.SLOT("map()"))
                mapper.setMapping(button,button_nums[j][k])
                tab_buttons[j].append(button)
        
        tab_button_desc = [[] for j in range(num_tabs)]
        for j in range(num_tabs):
            for k in range(num_elems[j]):
                tab_button_desc[j].append(QtGui.QLabel(button_desc[j][k]))
        
        tab_widgets=[[QtGui.QWidget() for k in range(num_elems[j])] for j in range(num_tabs)]
        tab_horiz_layouts=[[QtGui.QHBoxLayout(tab_widgets[j][k]) for k in range(num_elems[j])] for j in range(num_tabs)]
        for j in range(num_tabs):
            for k in range(num_elems[j]):
                tab_horiz_layouts[j][k].addWidget(tab_buttons[j][k])
                tab_horiz_layouts[j][k].addSpacing(20)
                tab_horiz_layouts[j][k].addWidget(tab_button_desc[j][k])
        for j in range(num_tabs):
            for k in range(num_elems[j]):
                tab_verts[j].addWidget(tab_widgets[j][k])
            tab_verts[j].addStretch()
        
        #set mapper to on_button_clicked funtions
        self.connect(mapper, QtCore.SIGNAL("mapped(int)"), self.on_button_clicked)
        
        
    
    def on_button_clicked(self,num):
        """
        Receives integers from button click to use for calling example script
        """
        print num
        #self.moveout()
        #exconfig.option=num
        #self.close()


class HoverButton(QtGui.QPushButton):
    def enterEvent(self,event):  
        self.setStyleSheet('QPushButton {font-family: Arial;border-width: 3px;border-color:#111111;border-style: solid;border-radius: 7;background: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #00DDDD, stop: 0.1 #00CCDD, stop: 0.49 #00BBDD, stop: 0.5 #00AADD, stop: 1 #0099DD)}')
    def leaveEvent(self,event):  
        self.setStyleSheet('QPushButton {font-family: Arial;border-width: 2px;border-color:#666666;border-style: solid;border-radius: 7;background: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #00DDDD, stop: 0.1 #00CCDD, stop: 0.49 #00BBDD, stop: 0.5 #00AADD, stop: 1 #0099DD)}')


class HoverExit(QtGui.QPushButton): 
    def enterEvent(self,event):  
        self.setStyleSheet('QPushButton {font-family: Arial;border-width: 3px;border-color:#111111;border-style: solid;border-radius: 7;font-size: 16px;background: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #FFAAAA, stop: 0.1 #FF9999, stop: 0.49 #FF8888, stop: 0.5 #FF7777, stop: 1 #FF6666)}')
    def leaveEvent(self,event):  
        self.setStyleSheet('QPushButton {font-family: Arial;border-width: 2px;border-color:#666666;border-style: solid;border-radius: 7;font-size: 16px;background: QLinearGradient( x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #FFAAAA, stop: 0.1 #FF9999, stop: 0.49 #FF8888, stop: 0.5 #FF7777, stop: 1 #FF6666)}')











