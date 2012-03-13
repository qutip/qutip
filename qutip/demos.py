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
from .examples import exconfig
from scipy import arange,array,any
import qutip.settings


def demos():
    """
    Calls the demos scripts via a GUI window if PySide
    or PyQt4 are avaliable.  Otherwise, a commandline 
    interface is given in the terminal.
    """
    exconfig.tab=0
    exconfig.button_num=0
    exconfig.is_green=0
    if qutip.settings.qutip_graphics=='YES':
        from gui import Examples
        if qutip.settings.qutip_gui=="PYSIDE":
            from PySide import QtGui, QtCore
        elif qutip.settings.qutip_gui=="PYQT4":
            from PyQt4 import QtGui, QtCore
        def start_gui(ver):
            app=QtGui.QApplication.instance()#checks if QApplication already exists (needed for iPython)
            if not app:#create QApplication if it doesnt exist
                app = QtGui.QApplication(sys.argv)
            gui=Examples(ver)
            gui.show()
            gui.activateWindow()
            gui.raise_()
            app.exec_()
    else:
        opts=array([123456,11,12,13,14,15,21,22,23,24,25,31,32,33,34,35,41,42,43,44,45])
        lopts=arange(len(opts))
    exconfig.option=0
    
    while exconfig.option<123456:
        exconfig.option=123456
        if qutip.settings.qutip_graphics=='YES':
            import _version
            if _version.release:
                ver=_version.short_version
            else:
                ver='HEAD'
            start_gui(ver)
        else:
            if sys.stdout.isatty():
                print('')
                print('\nQuTiP Example Scripts:')
                print('-----------------------')
                #first row
                print('[1]  Basic Obj operations')
                print('[2]  Operator & state usage examples')
                print('[3]  Tensor / partial trace usage')
                print('[4]  Wigner & Q dist. of Schrodinger cat-state')
                print('[5]  Squeezed state')
                #second row
                print('[6]  Steady-state cavity+qubit')
                print('[7]  Steady-state oscillator in thermal bath')
                print('[8]  Eseries example')
                print('[9]  Master equation: Rabi oscillations')
                print('[10] Master equation: Single-atom lasing')
                #third row
                print('[11] Density matrix metrics: Fidelity')
                print('[12] Propagator: Steady-state of driven sys.')
                print('[13] Heisenberg spin-chain (N=4)')
                print('[14] Correlations and spectrum')
                print('[15] Qubit decay on the Bloch sphere')
                #forth row
                print('[16] Monte-Carlo: cavity+qubit')
                print('[17] Monte-Carlo: trilinear Hamiltonian')
                print('[18] Monte-Carlo: thermal deviations')
                print('[19] Time-dependent H: Rabi oscillations')
                print('[20] Time-dependent H: LZ transistions')
                print('[0]  Exit...')
                wflag=0
                while wflag<3:
                    userinpt=raw_input("\nPick an example to run:")
                    try:
                        userinpt=int(userinpt)
                    except:
                        print('Invalid choice.  Please pick again.')
                        wflag+=1
                    else:
                        if any(userinpt==lopts):
                            exconfig.option=opts[userinpt]
                            break
                        else:
                            print('Invalid choice.  Please pick again.')
                            wflag+=1
                if wflag==3:
                    print('\nThird time was not a charm in your case.')
                    print('It seems you cannot pick a valid option...\n')
                    return
            else:
                raise TypeError('Demos must be run from the terminal if no GUI is avaliable.')
        #run selected example
        if not exconfig.option==123456:
            example_code = compile('examples.ex_'+str(exconfig.option)+'.run()', '<string>', 'exec')
            eval(example_code)
        
            
            
            
