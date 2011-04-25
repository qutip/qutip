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
from Tkinter import *
import numpy
import scipy
CD_BASE = os.path.dirname(__file__) # get directory of about.py file
execfile(os.path.join(CD_BASE, "_version.py")) #execute _version.py file in CD_BASE directory
def about():
    tk_conify_center()
    if sys.platform=='darwin':
        from mac import MsgBox
        import matplotlib
        Mversion = "Matplotlib Ver: "+matplotlib.__version__
        title='About'
        message='QuTIP: The Quantum Optics Toolbox in Python'
        info='Copyright (c) 2011\nPaul D. Nation & Robert J. Johansson \n\n'+'QuTIP Ver:        '+__version__+"\nNumpy Ver:      "+numpy.__version__+'\n'+"Scipy Ver:         "+scipy.__version__+'\n'+Mversion+"\n\nQuTiP is released under the GPL3.\nSee the enclosed COPYING.txt\nfile for more information."
        MsgBox(title,message,info)
    elif sys.platform=='linux2' and os.environ['QUTIP_GRAPHICS']=='YES':
        from linux import AboutBox
        AboutBox(__version__)
    elif os.environ['QUTIP_GRAPHICS']=='YES':
        root = Tk()
        root.title('      About')
        root.wm_attributes("-topmost", 1)
        root.focus()
        def center(window):
            sw = window.winfo_screenwidth()
            sh = window.winfo_screenheight()
            rw = window.winfo_reqwidth()
            rh = window.winfo_reqheight()
            xc = (sw - rw) / 2
            yc = (sh -rh) / 2
            window.geometry("+%d+%d" % (xc, yc))
            window.deiconify()         # Harmless if window is already visible
            
        content = Frame(root)
        namelbl = Label(content, text="QuTIP: The Quantum Optics Toolbox in Python")
        auth1 = Label(content,text="Paul D. Nation")
        auth2 = Label(content,text="Robert J. Johansson")
        by = Label(content,text="By: ")
        spacer1 = Label(content,text="")
        Qversion = Label(content,text="QuTIP Version:  "+__version__)
        Nversion = Label(content, text="Numpy Version:  "+numpy.__version__)
        Sversion = Label(content, text="Scipy Version:  "+scipy.__version__)
        try:
            import matplotlib
        except:
            Mversion = Label(content, text="Matplotlib Version: None")
        else:
            Mversion = Label(content, text="Matplotlib Version:  "+matplotlib.__version__)
        content.grid(column=0, row=0)
        namelbl.grid(column=6, row=0, columnspan=2)
        by.grid(column=6,row=1,columnspan=3)
        auth1.grid(column=6, row=4, columnspan=2)
        auth2.grid(column=6, row=5, columnspan=2)
        spacer1.grid(column=6,row=6,columnspan=2)
        Qversion.grid(column=6,row=7,columnspan=2)
        Nversion.grid(column=6,row=8,columnspan=2)
        Sversion.grid(column=6,row=9,columnspan=2)
        Mversion.grid(column=6,row=10,columnspan=2)

        root.after(0,center,root)    # Zero delay doesn't seem to bother it
        root.mainloop()
    else:
        print "QuTIP: The Quantum Optics Toolbox in Python"
        print "Copyright (c) 2011"
        print "Paul D. Nation & Robert J. Johansson"
        print "QuTIP Version:  "+__version__
        print "Numpy Version:  "+numpy.__version__
        try:
            import matplotlib
        except:
            print "Matplotlib Version: None"
        else:
            print "Matplotlib Version:  "+matplotlib.__version__

def tk_conify_center():
    import os
    try: os.environ['FRANCO']=='TRUE'
    except: return
    else:
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
            os.remove(os.getcwd()+'/.egg.gif')
            os.environ['FRANCO']='FALSE'
        root=Tkinter.Tk() 
        root.title('The Franco Easter Egg')
        root.wm_attributes("-topmost", 1)
        zf=zipfile.ZipFile(".Tk.egg.zip", "r")
        data=zf.extract('.egg.gif',os.getcwd(),pwd='lowfruit')
        c=Tkinter.Canvas(root,width=290, height=300) 
        p=Tkinter.PhotoImage(file=data) 
        i=c.create_image(0,0,anchor=Tkinter.NW,image=p) 
        c.pack() 
        root.after(0,center,root)
        root.after(5000,stop,root)    
        root.mainloop()
        try:os.remove(os.getcwd()+'/.egg.gif')
        except:os.environ['FRANCO']='FALSE'
        else:os.environ['FRANCO']='FALSE'


if __name__ == "__main__":
    os.environ['QUTIP_GRAPHICS']='YES'
    about()
