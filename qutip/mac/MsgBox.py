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
import os
CD_BASE = os.path.dirname(__file__)
class MsgBox:
    """Simple class for displaying a message box using CocoaDialog"""

    # Change CD_BASE to reflect the location of Cocoadialog on your system
    CD_BASE = os.path.dirname(__file__)
    CD_PATH = os.path.join(CD_BASE, "CocoaDialog.app/Contents/MacOS/CocoaDialog")
    
    def __init__(self, title="Dialog", message="", info="", button1='close', icon='None'):
        """Create message box dialog"""
        template = "%s msgbox --title '%s' --text '%s' --informative-text '%s' --button1 '%s' --icon-file '%s' --float"
        self.pipe = os.popen(template % (MsgBox.CD_PATH, title, message, info, button1, icon), "w")
     	
       


if __name__ == "__main__":
    # Sample usage
    import time
    box = MsgBox(title="About",message="QuTIP: The Quantum Optics Toolbox in Python",info="informative-text goes here",button1="close")