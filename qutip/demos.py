#This file is part of QuTiP.
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
import sys,os
import qutip.examples as examples
from qutip.examples import exconfig
from qutip.examples.examples_text import button_labels,button_nums
from scipy import arange,array,any
import qutip.settings


def demos():
    """
    Calls the demos scripts via a GUI window if PySide
    or PyQt4 are avaliable.  Otherwise, a commandline 
    interface is given in the terminal.
    """
    direc=os.path.dirname(__file__)
    exconfig.tab=0
    exconfig.button_num=0
    exconfig.is_green=0
    exconfig.cmd_screen=1
    if qutip.settings.qutip_gui!='NONE':
        from gui import Examples
        if qutip.settings.qutip_gui=="PYSIDE":
            from PySide import QtGui, QtCore
        elif qutip.settings.qutip_gui=="PYQT4":
            from PyQt4 import QtGui, QtCore
        def start_gui(ver,direc):
            app=QtGui.QApplication.instance()#checks if QApplication already exists (needed for iPython)
            if not app:#create QApplication if it doesnt exist
                app = QtGui.QApplication(sys.argv)
            gui=Examples(ver,direc)
            gui.show()
            gui.activateWindow()
            gui.raise_()
            app.exec_()
    else:
        opts=array([button_nums[k] for k in range(len(button_nums))])
        lopts=arange(len(opts))
    exconfig.option=0
    
    while exconfig.option<123456:
        exconfig.option=123456
        if qutip.settings.qutip_gui!='NONE':
            import _version
            if _version.release:
                ver=_version.short_version
            else:
                ver='HEAD'
            start_gui(ver,direc)
            if not exconfig.option==123456:
                example_code = compile('examples.ex_'+str(exconfig.option)+'.run()', '<string>', 'exec')
                eval(example_code)
        else:
            #---Commandline Demos output---#
            if sys.stdout.isatty():
                while exconfig.cmd_screen!=0:
                    bnums=button_nums[exconfig.cmd_screen-1]
                    blabels=button_labels[exconfig.cmd_screen-1]
                    print("\n"*5)
                    #first screen
                    if exconfig.cmd_screen==1:
                        print('\nQuTiP Basic Example Scripts:')
                        print('=============================')
                        for jj in range(len(bnums)):
                            print("["+str(bnums[jj])+"] "+blabels[jj])
                        print('[1] Next Page ==>')
                        print('[0] Exit Demos')
                    #last screen
                    elif exconfig.cmd_screen==5:
                        print('\nQuTiP Advanced Example Scripts:')
                        print('================================')
                        for jj in range(len(bnums)):
                            print("["+str(bnums[jj])+"] "+blabels[jj])
                        print('[2] Previous Page <==')
                        print('[0] Exit Demos')
                    #in between screens
                    else:
                        tt=["Master Equation","Monte Carlo","Time-Dependent"]
                        print("\nQuTiP "+tt[exconfig.cmd_screen-2]+" Example Scripts:")
                        print('======================================')
                        for jj in range(len(bnums)):
                            print("["+str(bnums[jj])+"] "+blabels[jj])
                        print('[1] Next Page ==>')
                        print('[2] Previous Page <==')
                        print('[0] Exit Demos')
                    #code for selecting examples
                    wflag=0
                    while wflag<3:
                        userinpt=raw_input("\nPick an example to run:")
                        try:
                            userinpt=int(userinpt)
                        except:
                            print('Invalid choice.  Please pick again.')
                            wflag+=1
                        else:
                            if userinpt==0:
                                exconfig.cmd_screen=0
                                exconfig.option=123456
                                break
                            elif userinpt==1:
                                if exconfig.cmd_screen==5:
                                    pass
                                else:
                                    exconfig.cmd_screen+=1
                                exconfig.option=123456
                                break
                            elif userinpt==2:
                                if exconfig.cmd_screen==1:
                                    pass
                                else:
                                    exconfig.cmd_screen-=1
                                exconfig.option=123456
                                break 
                            elif any(userinpt==opts[exconfig.cmd_screen-1]):
                                exconfig.option=userinpt
                                break
                            else:
                                print('Invalid choice.  Please pick again.')
                                wflag+=1
                    if wflag==3:
                        print('\nThird time was not a charm in your case.')
                        print('It seems you cannot pick a valid option...\n')
                        return
                    if not exconfig.option==123456:
                        example_code = compile('examples.ex_'+str(exconfig.option)+'.run()', '<string>', 'exec')
                        eval(example_code)
                    
            else:
                #raise exception if running demos from scipt with no GUI.
                raise Exception('Demos must be run from the terminal if no GUI is avaliable.')
        
        
        
            
            
            
