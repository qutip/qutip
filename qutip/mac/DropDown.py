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

class DropDown:
    """Simple class for displaying a message box using CocoaDialog"""

    # Change CD_BASE to reflect the location of Cocoadialog on your system
    CD_BASE = os.path.dirname(__file__)
    CD_PATH = os.path.join(CD_BASE, "CocoaDialog.app/Contents/MacOS/CocoaDialog")
    
    def __init__(self, title="", items='', button1="Run", button2="Cancel"):
        """Create message box dialog"""
        template = "%s dropdown --title '%s' --items %s --button1 '%s' --button2 '%s' --float"
        self.pipe = os.popen(template % (DropDown.CD_PATH, title, items, button1, button2), "w")
     	
       


if __name__ == "__main__":
    # Sample usage
	box = DropDown(title="QuTIP Examples",items="'Partial Trace (xptrace)' 'Schrodinger Cat (xschcat)' 'Steady-State (xprobss)'",button1="Run", button2="Cancel")