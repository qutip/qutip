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

import pygtk
pygtk.require('2.0')
import gtk
import gobject
import time


class ProgressBar:

    def update(self, percent, message=False):
    
        if message:
            self.pbar.set_text(" " + message + " ")       

        self.pbar.set_fraction(percent / 100.0)

        while gtk.events_pending():
            gtk.main_iteration(False)

    def destroy(self, widget, data=None):
        self.finish()

    def finish(self):
        self.window.destroy()
        #gtk.main_quit()

    def hide(self, widget):
        self.window.hide()

    def __init__(self, title):
        self.window = gtk.Window(gtk.WINDOW_TOPLEVEL)
        self.window.set_resizable(True)
        self.window.set_position(gtk.WIN_POS_CENTER)
        self.window.connect("destroy", self.destroy)
        self.window.set_title("QuTiP: " + title)
        self.window.set_border_width(0)

        vbox = gtk.VBox(False, 5)
        vbox.set_border_width(10)
        self.window.add(vbox)
        vbox.show()
  
        align = gtk.Alignment(0.5, 0.5, 0, 0)
        vbox.pack_start(align, True, False, 5)
        align.show()
        self.pbar = gtk.ProgressBar()
        align.add(self.pbar)
        self.pbar.set_size_request(400, 30)
        # optionally set text in progress bar
        self.title = title
        if self.title:
            self.pbar.set_text(" " + self.title + " ")

        self.pbar.show()

        separator = gtk.HSeparator()
        vbox.pack_start(separator, False, False, 0)
        separator.show()

        #button = gtk.Button("Hide")
        #button.connect("clicked", self.hide)
        #vbox.pack_start(button, False, False, 0)
        #button.set_flags(gtk.CAN_DEFAULT)
        #button.grab_default()
        #button.show()

        self.window.show()
        
            



if __name__ == "__main__":

    ntraj = 100

    pbar = ProgressBar(title="Running Monte-Carlo Trajectories")
    for percent in range(1,ntraj+1):
        time.sleep(.1)
        #pbar.update(float(percent)/ntraj)
        pbar.update(float(percent), "Trajectories completed: " + str(percent) + "/" + str(ntraj))
    time.sleep(.3)
    pbar.finish()
    print "done"

