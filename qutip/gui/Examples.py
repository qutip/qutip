import qutip.examples
import sys,os,time
from numpy import arange
from qutip.examples import exconfig

if os.environ['QUTIP_GUI']=="PYSIDE":
    from PySide import QtGui, QtCore

elif os.environ['QUTIP_GUI']=="PYQT4":
    from PyQt4 import QtGui, QtCore



#basic demos
_basic_demos_labels=["Schrodingers Cat","Q-function","Qobj Eigenvalues/Eigenvectors","blank","blank","blank","blank"]

_basic_demos_descriptions=["Schrodingers Cat state from \nsuperposition of two coherent states.",
                            "Q-function from superposition \nof two coherent states.",
                            "Eigenvalues/Eigenvectors of cavity-qubit system \nin strong-coupling regime.",
                            "Bloch Sphere","blank","blank","blank"]

_basic_output_nums=arange(1,len(_basic_demos_labels)+1) #does not start at zero so commandline output numbers match (0=quit in commandline)

#master equation demos
_master_demos_labels=["blank","blank","blank","blank","blank","blank","blank"]
_master_demos_descriptions=["blank","blank","blank","blank","blank","blank","blank"]
_master_output_nums=10+arange(len(_master_demos_labels))



_monte_demos_labels=["blank","blank","blank","blank","blank","blank","blank"]
_monte_demos_descriptions=["blank","blank","blank","blank","blank","blank","blank"]
_monte_output_nums=20+arange(len(_monte_demos_labels))


_td_demos_labels=["blank","blank","blank","blank","blank","blank","blank"]
_td_demos_descriptions=["blank","blank","blank","blank","blank","blank","blank"]
_td_output_nums=30+arange(len(_td_demos_labels))

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
        quit.setStyleSheet('QPushButton {font-family: Arial;border-width: 2px;border-color:#666666;border-style: solid;border-radius: 7;font-size: 16px;}')
        quit.setFocusPolicy(QtCore.Qt.NoFocus)
        quit.clicked.connect(self.close)
        
        #tab widgets
        tab_widget = QtGui.QTabWidget(self) 
        tab_widget.move(10,100)
        tab_widget.resize(770,500)
        tab1 = QtGui.QWidget(self)
        tab_widget.addTab(tab1, "Basic Operations")
        tab2 = QtGui.QWidget(self)
        tab_widget.addTab(tab2, "Master Equation")
        tab3 = QtGui.QWidget(self)
        tab_widget.addTab(tab3, "Monte Carlo")
        tab4 = QtGui.QWidget(self)
        tab_widget.addTab(tab4, "Time-Dependent")
        
        tab_button_style='QPushButton {font-family: Arial;border-width: 2px;border-color:#666666;border-style: solid;border-radius: 7}'
        
        
        #tab 1 buttons
        tab1_vert = QtGui.QVBoxLayout(tab1)
        num_elem1=len(_basic_demos_labels)
        tab1_button_descs = [QtGui.QLabel(_basic_demos_descriptions[k]) for k in range(num_elem1)]
        tab1_buttons = [HoverButton() for k in range(num_elem1)]
        for k in range(num_elem1):
            tab1_buttons[k].setFont(font)
            tab1_buttons[k].setText(_basic_demos_labels[k])
            tab1_buttons[k].setStyleSheet(tab_button_style)
            tab1_buttons[k].setFixedSize(335, 40)
            self.connect(tab1_buttons[k], QtCore.SIGNAL("clicked()"), mapper, QtCore.SLOT("map()"))
            mapper.setMapping(tab1_buttons[k], _basic_output_nums[k])
        tab1_example_widgets=[QtGui.QWidget() for k in range(num_elem1)]
        tab1_hori_layouts=[QtGui.QHBoxLayout(tab1_example_widgets[k]) for k in range(num_elem1)]
        for k in range(num_elem1):
            tab1_hori_layouts[k].addWidget(tab1_buttons[k])
            tab1_hori_layouts[k].addSpacing(20)
            tab1_hori_layouts[k].addWidget(tab1_button_descs[k])
        for k in range(num_elem1):
            tab1_vert.addWidget(tab1_example_widgets[k])
        tab1_vert.addStretch()
        
        #tab 2 buttons
        tab2_vert = QtGui.QVBoxLayout(tab2)
        num_elem2=len(_master_demos_labels)
        tab2_button_descs = [QtGui.QLabel(_master_demos_descriptions[k]) for k in range(num_elem2)]
        tab2_buttons = [HoverButton() for k in range(num_elem2)]
        for k in range(num_elem2):
            tab2_buttons[k].setFont(font)
            tab2_buttons[k].setText(_master_demos_labels[k])
            tab2_buttons[k].setStyleSheet(tab_button_style)
            tab2_buttons[k].setFixedSize(335, 40)
            self.connect(tab2_buttons[k], QtCore.SIGNAL("clicked()"), mapper, QtCore.SLOT("map()"))
            mapper.setMapping(tab2_buttons[k], _master_output_nums[k])
        tab2_example_widgets=[QtGui.QWidget() for k in range(num_elem2)]
        tab2_hori_layouts=[QtGui.QHBoxLayout(tab2_example_widgets[k]) for k in range(num_elem2)]
        for k in range(num_elem2):
            tab2_hori_layouts[k].addWidget(tab2_buttons[k])
            tab2_hori_layouts[k].addSpacing(20)
            tab2_hori_layouts[k].addWidget(tab2_button_descs[k])
        for k in range(num_elem2):
            tab2_vert.addWidget(tab2_example_widgets[k])
        tab2_vert.addStretch()
        
        #tab 3 buttons
        tab3_vert = QtGui.QVBoxLayout(tab3)
        num_elem3=len(_monte_demos_labels)
        tab3_button_descs = [QtGui.QLabel(_monte_demos_descriptions[k]) for k in range(num_elem3)]
        tab3_buttons = [HoverButton() for k in range(num_elem3)]
        for k in range(num_elem3):
            tab3_buttons[k].setFont(font)
            tab3_buttons[k].setText(_monte_demos_labels[k])
            tab3_buttons[k].setStyleSheet(tab_button_style)
            tab3_buttons[k].setFixedSize(335, 40)
            self.connect(tab3_buttons[k], QtCore.SIGNAL("clicked()"), mapper, QtCore.SLOT("map()"))
            mapper.setMapping(tab3_buttons[k], _monte_output_nums[k])
        tab3_example_widgets=[QtGui.QWidget() for k in range(num_elem3)]
        tab3_hori_layouts=[QtGui.QHBoxLayout(tab3_example_widgets[k]) for k in range(num_elem3)]
        for k in range(num_elem3):
            tab3_hori_layouts[k].addWidget(tab3_buttons[k])
            tab3_hori_layouts[k].addSpacing(20)
            tab3_hori_layouts[k].addWidget(tab3_button_descs[k])
        for k in range(num_elem3):
            tab3_vert.addWidget(tab3_example_widgets[k])
        tab3_vert.addStretch()
        
        #tab 4 buttons
        tab4_vert = QtGui.QVBoxLayout(tab4)
        num_elem4=len(_td_demos_labels)
        tab4_button_descs = [QtGui.QLabel(_td_demos_descriptions[k]) for k in range(num_elem4)]
        tab4_buttons = [HoverButton() for k in range(num_elem4)]
        for k in range(num_elem4):
            tab4_buttons[k].setFont(font)
            tab4_buttons[k].setText(_td_demos_labels[k])
            tab4_buttons[k].setStyleSheet(tab_button_style)
            tab4_buttons[k].setFixedSize(335, 40)
            self.connect(tab4_buttons[k], QtCore.SIGNAL("clicked()"), mapper, QtCore.SLOT("map()"))
            mapper.setMapping(tab4_buttons[k],_td_output_nums[k])
        tab4_example_widgets=[QtGui.QWidget() for k in range(num_elem4)]
        tab4_hori_layouts=[QtGui.QHBoxLayout(tab4_example_widgets[k]) for k in range(num_elem4)]
        for k in range(num_elem4):
            tab4_hori_layouts[k].addWidget(tab4_buttons[k])
            tab4_hori_layouts[k].addSpacing(20)
            tab4_hori_layouts[k].addWidget(tab4_button_descs[k])
        for k in range(num_elem4):
            tab4_vert.addWidget(tab4_example_widgets[k])
        tab4_vert.addStretch()
        
        
        self.connect(mapper, QtCore.SIGNAL("mapped(int)"), self.on_button)
        
        
    
    def on_button(self,num):
        """
        Receives integers from button click to use for calling example script
        """
        print num
        #self.moveout()
        #exconfig.option=num
        #self.close()


class HoverButton(QtGui.QPushButton):
    def enterEvent(self,event):  
        self.setStyleSheet('QPushButton {background: #888888;font-family: Arial;border-width: 2px;border-color:#222222;border-style: solid;border-radius: 7}')
    def leaveEvent(self,event):  
        self.setStyleSheet('QPushButton {font-family: Arial;border-width: 2px;border-color:#666666;border-style: solid;border-radius: 7}')


class HoverExit(QtGui.QPushButton): 
    def enterEvent(self,event):  
        self.setStyleSheet('QPushButton {background: #888888;font-family: Arial;border-width: 2px;border-color:#222222;border-style: solid;border-radius: 7;font-size: 16px;}')
    def leaveEvent(self,event):  
        self.setStyleSheet('QPushButton {font-family: Arial;border-width: 2px;border-color:#666666;border-style: solid;border-radius: 7;font-size: 16px;}')











