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
import sys,os
import numpy,scipy
CD_BASE = os.path.dirname(__file__) # get directory of about.py file
execfile(os.path.join(CD_BASE, "_version.py")) #execute _version.py file in CD_BASE directory
def about():
    """
    About box for QuTiP.  Gives version numbers for 
    QuTiP, NumPy, SciPy, and MatPlotLib.
    
    GUI version requires PySide or PyQt4.
    """
    tk_conify_center()
    if os.environ['QUTIP_GRAPHICS']=='YES':
        from gui import AboutBox
        import matplotlib
        if os.environ['QUTIP_GUI']=="PYSIDE":
            from PySide import QtGui, QtCore
        elif os.environ['QUTIP_GUI']=="PYQT4":
            from PyQt4 import QtGui, QtCore
        
        app=QtGui.QApplication.instance()#checks if QApplication already exists (needed for iPython)
        if not app:#create QApplication if it doesnt exist
            app = QtGui.QApplication(sys.argv)
        box=AboutBox(version)
        box.show()
        box.raise_()
        app.exec_()
        
        
        
    else:
        import matplotlib
        print "QuTIP: The Quantum Optics Toolbox in Python"
        print "Copyright (c) 2011"
        print "Paul D. Nation & Robert J. Johansson"
        print "QuTIP Version:  "+version
        print "Numpy Version:  "+numpy.__version__
        print "Scipy Version:  "+scipy.__version__
        print "Matplotlib Version:  "+matplotlib.__version__
        print ''

def tk_conify_center():
    import os
    try: os.environ['FRANCO']
    except: pass
    else:
        if os.environ['FRANCO']=='TRUE':
            import Tkinter,zipfile,time
            def center(window):
              sw = window.winfo_screenwidth()
              sh = window.winfo_screenheight()
              rw = window.winfo_reqwidth()
              rh = window.winfo_reqheight()
              xc = (sw - rw) / 2
              yc = (sh -rh) / 2
              window.geometry("+%d+%d" % (xc-75, yc-75))
              window.deiconify()
            def stop(me):
                stop_flag=1
                me.destroy()
                os.remove(os.path.dirname(__file__)+'/.egg.gif')
                os.environ['FRANCO']='FALSE'
            root=Tkinter.Tk() 
            root.title('The Franco Easter Egg')
            root.wm_attributes("-topmost", 1)
            zf=zipfile.ZipFile(os.path.dirname(__file__)+"/.Tk.egg.zip", 'r')
            data=zf.extract('.egg.gif',os.path.dirname(__file__),pwd='lowfruit')
            c=Tkinter.Canvas(root,width=290, height=300) 
            p=Tkinter.PhotoImage(file=data) 
            i=c.create_image(0,0,anchor=Tkinter.NW,image=p) 
            c.pack() 
            root.after(0,center,root)
            root.after(5000,stop,root)    
            root.mainloop()
            try:os.remove(os.path.dirname(__file__)+'/.egg.gif')
            except:os.environ['FRANCO']='FALSE'
            else:os.environ['FRANCO']='FALSE'


if __name__ == "__main__":
    os.environ['QUTIP_GRAPHICS']='NO'
    #os.environ['QUTIP_GUI']='PYSIDE'
    about()
    about()
