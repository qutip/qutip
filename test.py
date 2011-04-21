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
###########################################################################
from qutip import *
import sys,time
###-platform dependent imports-###
if sys.platform=='darwin':
    from qutip.mac import ProgressBar
else:
    from qutip.linux import Tk_ProgressBar,center
    import Tkinter
#---------------------------------

class Counter():
    def __init__(self,max,step_size=1):
        self.max=max
        self.count=0
        self.step=step_size
        self.percent=0.0
        if sys.platform=='darwin':
            self.bar = ProgressBar(title="Running Monte-Carlo Trajectories:")
        else:
            self.root = Tkinter.Tk(className=' Running Monte-Carlo Trajectories:')
            self.root.withdraw()
            self.root.focus()
            self.root.after(0,center,self.root)
            self.bar=Tk_ProgressBar(self.root, relief='ridge', bd=3)
            self.bar.pack(fill='x')
            self.bar.set(self.percent, str(self.percent*100)+' %')
                
    def update(self):#update the counter
        self.count+=self.step
        self.percent=self.count/(1.0*self.max)
        if sys.platform=='darwin':
            self.bar.update(self.percent*100.0, "Trajectories completed: "+str(self.count)+"/"+str(self.max))
        else:
            self.bar.set(self.percent,"Trajectories completed: "+str(self.count)+"/"+str(self.max))
    
    def finish(self):#pause and close and finish
        time.sleep(0.5)
        if sys.platform=='darwin':
            self.bar.finish()
        else:
            self.root.destroy()


######---Demo---######
if __name__ == "__main__":
    x=Counter(100)
    def func(x):
        for i in range(100):
            time.sleep(.1)
            x.update()
        time.sleep(.5)
        x.finish()
    

    if sys.platform!='darwin':
        x.bar.after(0,lambda: func(x))
        x.root.mainloop()
    else:
        func(x)
    print 'finished'









           