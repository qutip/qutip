# This file is part of QuTiP.
#
#    QuTiP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    QuTiP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTiP.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright (C) 2011 and later, Paul D. Nation & Robert J. Johansson
#
###########################################################################
import os,sys
import qutip
if os.environ['QUTIP_GUI'] == "PYSIDE":
    from PySide import QtGui, QtCore

elif os.environ['QUTIP_GUI'] == "PYQT4":
    from PyQt4 import QtGui, QtCore

import numpy
import scipy
import matplotlib
import Cython
from qutip import _version2int, __version__ as qutip_version

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

CD_BASE = os.path.dirname(__file__)

class Aboutbox(object):
    def setupUi(self, Form):
        Form.setObjectName(_fromUtf8("About QuTiP"))
        Form.resize(365, 505)

        self.gridLayout = QtGui.QGridLayout(Form)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))

        self.label = QtGui.QLabel(Form)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(18)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setIndent(0)
        self.label.setObjectName(_fromUtf8("label"))

        self.gridLayout.addWidget(self.label, 2, 0, 1, 1)
        spacerItem = QtGui.QSpacerItem(20, 5, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem, 3, 0, 1, 1)
        spacerItem1 = QtGui.QSpacerItem(20, 5, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.gridLayout.addItem(spacerItem1, 1, 0, 1, 1)

        self.label_4 = QtGui.QLabel(Form)
        self.label_4.setMinimumSize(QtCore.QSize(240, 0))
        self.label_4.setMaximumSize(QtCore.QSize(16777215, 150))
        self.label_4.setText(_fromUtf8(""))
        self.label_4.setPixmap(QtGui.QPixmap(_fromUtf8(CD_BASE + "/logo.png")))
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setIndent(0)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.gridLayout.addWidget(self.label_4, 0, 0, 1, 1)

        self.label_3 = QtGui.QLabel(Form)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Arial"))
        font.setPointSize(13)
        font.setBold(False)
        font.setWeight(50)
        self.label_3.setFont(font)
        self.label_3.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.gridLayout.addWidget(self.label_3, 6, 0, 1, 1)

        self.tabWidget = QtGui.QTabWidget(Form)
        self.tabWidget.setObjectName(_fromUtf8("tabWidget"))
        self.tab_vers = QtGui.QWidget()
        self.tab_vers.setObjectName(_fromUtf8("tab_vers"))
        self.tabWidget.addTab(self.tab_vers, _fromUtf8(""))

        # Version info tab content
        font1 = QtGui.QFont()
        font1.setFamily(_fromUtf8("Arial"))
        font1.setPointSize(16)
        font1.setBold(True)
        font1.setWeight(75)

        font2 = QtGui.QFont()
        font2.setFamily(_fromUtf8("Arial"))
        font2.setPointSize(16)
        font2.setBold(False)
        font2.setWeight(50)

        self.version_info_grid = QtGui.QGridLayout(self.tab_vers)

        # qutip
        self.qutip_label = QtGui.QLabel()
        self.qutip_label.setFont(font1)
        self.qutip_label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.qutip_label.setObjectName(_fromUtf8("qutip_label"))
        self.version_info_grid.addWidget(self.qutip_label, 0, 0)

        self.qutip_version = QtGui.QLabel()
        self.qutip_version.setFont(font2)
        self.qutip_version.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.qutip_version.setObjectName(_fromUtf8("qutip_version"))
        self.version_info_grid.addWidget(self.qutip_version, 0, 1)

        # numpy
        self.numpy_label = QtGui.QLabel()
        self.numpy_label.setFont(font1)
        self.numpy_label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.numpy_label.setObjectName(_fromUtf8("numpy_label"))
        self.version_info_grid.addWidget(self.numpy_label, 1, 0)

        self.numpy_version = QtGui.QLabel()
        self.numpy_version.setFont(font2)
        self.numpy_version.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.numpy_version.setObjectName(_fromUtf8("numpy_version"))
        self.version_info_grid.addWidget(self.numpy_version, 1, 1)

        # scipy
        self.scipy_label = QtGui.QLabel()
        self.scipy_label.setFont(font1)
        self.scipy_label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.scipy_label.setObjectName(_fromUtf8("scipy_label"))
        self.version_info_grid.addWidget(self.scipy_label, 2, 0)

        self.scipy_version = QtGui.QLabel()
        self.scipy_version.setFont(font2)
        self.scipy_version.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.scipy_version.setObjectName(_fromUtf8("scipy_version"))
        self.version_info_grid.addWidget(self.scipy_version, 2, 1)

        # cython
        self.cython_label = QtGui.QLabel()
        self.cython_label.setFont(font1)
        self.cython_label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.cython_label.setObjectName(_fromUtf8("cython_label"))
        self.version_info_grid.addWidget(self.cython_label, 3, 0) 

        self.cython_version = QtGui.QLabel()
        self.cython_version.setFont(font2)
        self.cython_version.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.cython_version.setObjectName(_fromUtf8("cython_version"))
        self.version_info_grid.addWidget(self.cython_version, 3, 1) 

        # matplotlib
        self.mpl_label = QtGui.QLabel()
        self.mpl_label.setFont(font1)
        self.mpl_label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.mpl_label.setObjectName(_fromUtf8("mpl_label"))
        self.version_info_grid.addWidget(self.mpl_label, 4, 0) 

        self.mpl_version = QtGui.QLabel()
        self.mpl_version.setFont(font2)
        self.mpl_version.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.mpl_version.setObjectName(_fromUtf8("mpl_version"))
        self.version_info_grid.addWidget(self.mpl_version, 4, 1) 

        # pyside
        self.pyside_label = QtGui.QLabel()
        self.pyside_label.setFont(font1)
        self.pyside_label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.pyside_label.setObjectName(_fromUtf8("pyside_label"))
        self.version_info_grid.addWidget(self.pyside_label, 5, 0) 

        self.pyside_version = QtGui.QLabel()
        self.pyside_version.setFont(font2)
        self.pyside_version.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.pyside_version.setObjectName(_fromUtf8("pyside_version"))
        self.version_info_grid.addWidget(self.pyside_version, 5, 1) 

        # pyqt4
        self.pyqt4_label = QtGui.QLabel()
        self.pyqt4_label.setFont(font1)
        self.pyqt4_label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.pyqt4_label.setObjectName(_fromUtf8("pyqt4_label"))
        self.version_info_grid.addWidget(self.pyqt4_label, 6, 0) 

        self.pyqt4_version = QtGui.QLabel()
        self.pyqt4_version.setFont(font2)
        self.pyqt4_version.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.pyqt4_version.setObjectName(_fromUtf8("pyqt4_version"))
        self.version_info_grid.addWidget(self.pyqt4_version, 6, 1) 

        # pyobjc
        if sys.platform == 'darwin':
            self.pyobjc_label = QtGui.QLabel()
            self.pyobjc_label.setFont(font1)
            self.pyobjc_label.setLayoutDirection(QtCore.Qt.LeftToRight)
            self.pyobjc_label.setObjectName(_fromUtf8("pyobjc_label"))
            self.version_info_grid.addWidget(self.pyobjc_label, 7, 0) 

            self.pyobjc_version = QtGui.QLabel()
            self.pyobjc_version.setFont(font2)
            self.pyobjc_version.setLayoutDirection(QtCore.Qt.LeftToRight)
            self.pyobjc_version.setObjectName(_fromUtf8("pyobjc_version"))
            self.version_info_grid.addWidget(self.pyobjc_version, 7, 1) 


        # Developers tab
        self.tab_devs = QtGui.QWidget()
        self.tab_devs.setObjectName(_fromUtf8("tab_devs"))
        self.tabWidget.addTab(self.tab_devs, _fromUtf8(""))

        self.devs_grid = QtGui.QGridLayout(self.tab_devs)

        self.devs_title = QtGui.QLabel()
        self.devs_title.setFont(font1)
        self.devs_title.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.devs_title.setObjectName(_fromUtf8("devs_title"))
        self.devs_grid.addWidget(self.devs_title, 0, 0) 

        self.devs = QtGui.QLabel()
        self.devs.setFont(font2)
        self.devs.setOpenExternalLinks(True)
        self.devs.setObjectName(_fromUtf8("devs"))
        self.devs_grid.addWidget(self.devs, 1, 0) 

        self.contribs_title = QtGui.QLabel()
        self.contribs_title.setFont(font1)
        self.contribs_title.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.contribs_title.setObjectName(_fromUtf8("contribs_title"))
        self.devs_grid.addWidget(self.contribs_title, 2, 0) 

        self.contribs = QtGui.QLabel()
        self.contribs.setFont(font2)
        self.contribs.setObjectName(_fromUtf8("contribs"))
        self.devs_grid.addWidget(self.contribs, 3, 0) 

        self.docs = QtGui.QLabel()
        self.docs.setFont(font2)
        self.docs.setOpenExternalLinks(True)
        self.docs.setObjectName(_fromUtf8("docs"))
        self.devs_grid.addWidget(self.docs, 4, 0) 

        self.gridLayout.addWidget(self.tabWidget, 5, 0, 1, 1)
        self.retranslateUi(Form)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        from qutip import version
        if version.release:
            version = version.short_version
        else:
            version = 'HEAD'
        try:
            import PySide
            pyside_ver = PySide.__version__
        except:
            pyside_ver = 'None'
        try:
            import PyQt4.QtCore as qt4Core
            pyqt4_ver = qt4Core.PYQT_VERSION_STR
        except:
            pyqt4_ver = 'None'

        if sys.platform == 'darwin':
            try:
                import Foundation
                pyobjc = 'Yes'
            except:
                pyobjc = 'No'

        Form.setWindowTitle(_translate("Form", "About QuTiP", None))
        Form.setWindowIcon(QtGui.QIcon(CD_BASE + "/logo.png"))

        self.label.setText(_translate("Form", "QuTiP: The Quantum Toolbox in Python", None))
        self.label_3.setText(_translate("Form", "Copyright 2011 and later, P. D. Nation & J. R. Johansson", None))

        self.qutip_label.setText(_translate("Form", "QuTiP Version:", None))
        self.qutip_version.setText(_translate("Form", qutip_version, None))

        self.numpy_label.setText(_translate("Form", "NumPy Version:", None))
        self.numpy_version.setText(_translate("Form", str(numpy.__version__), None))

        self.scipy_label.setText(_translate("Form", "SciPy Version:", None))
        self.scipy_version.setText(_translate("Form", str(scipy.__version__), None))

        self.cython_label.setText(_translate("Form", "Cython Version:", None))
        self.cython_version.setText(_translate("Form", str(Cython.__version__), None))

        self.mpl_label.setText(_translate("Form", "Matplotlib Version:", None))
        self.mpl_version.setText(_translate("Form", str(matplotlib.__version__), None))

        self.pyside_label.setText(_translate("Form", "PySide Version:", None))
        self.pyside_version.setText(_translate("Form", pyside_ver, None))

        self.pyqt4_label.setText(_translate("Form", "PyQt4 Version:", None))
        self.pyqt4_version.setText(_translate("Form", pyqt4_ver, None))

        if sys.platform == 'darwin':
            self.pyobjc_label.setText(_translate("Form", "PyObjC Installed:", None))
            self.pyobjc_version.setText(_translate("Form", str(pyobjc), None))

        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_vers),
                                  _translate("Form", "Version Info", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_devs),
                                  _translate("Form", "Developers", None))

        self.devs_title.setText(_translate("Form", "Lead Developers:", None))
        self.devs.setText(_translate("Form", "<a href=\"http://dml.riken.jp/~rob\"><span style=\" text-decoration: underline; color:#0000ff;\">Robert Johansson</span></a> & <a href=\"http://nqdl.korea.ac.kr\"><span style=\" text-decoration: underline; color:#0000ff;\">Paul Nation</span></a>", None))

        self.contribs_title.setText(_translate("Form", "Contributors:", None))
        self.contribs.setText(_translate("Form", "Arne Grimsmo, Markus Baden", None))
        self.docs.setText(_translate("Form", "For a list of bug hunters and other supporters, <br />see the <a href=\"http://qutip.googlecode.com/svn/doc/"+qutip_version[0:5]+"/html/index.html\"><span style=\" text-decoration: underline; color:#0000ff;\">QuTiP Documentation</span></a>", None))


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    Form = QtGui.QWidget()
    ui = Aboutbox()
    ui.setupUi(Form)
    Form.activateWindow()
    Form.setFocus()
    Form.show()
    Form.raise_()
    app.exec_()

