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

class ProgressBar:
    """Simple class for displaying progress bars using CocoaDialog"""

    # Change CD_BASE to reflect the location of Cocoadialog on your system
    CD_BASE = os.path.dirname(__file__)
    CD_PATH = os.path.join(CD_BASE, "CocoaDialog.app/Contents/MacOS/CocoaDialog")
    
    def __init__(self, title="Progress", message="", percent=0):
        """Create progress bar dialog"""
        template = "%s progressbar --title '%s' --text '%s' --percent %d"
        self.percent = percent
        self.pipe = os.popen(template % (ProgressBar.CD_PATH, title, message, percent), "w")
        self.message = message
            
    def update(self, percent, message=False):
        """Update progress bar (and message if desired)"""
        if message:
            self.message = message  # store message for persistence
        self.pipe.write("%d %s\n" % (percent, self.message))
        self.pipe.flush()
        
    def finish(self):
        """Close progress bar window"""
        self.pipe.close()


if __name__ == "__main__":
    # Sample usage
    import time
    bar = ProgressBar(title="Running Monte-Carlo Trajectories:")
    
    for percent in range(25):
        time.sleep(.2)
        bar.update(percent, "Trajectories completed: "+str(percent)+"/10000")
        
    for percent in range(25,100):
        time.sleep(.02)
        bar.update(percent, "Trajectories completed: "+str(percent)+"/10000")
     
    time.sleep(.75)
    bar.finish()