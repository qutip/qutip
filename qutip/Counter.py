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
import sys,time,os
from scipy import floor

# default, if __init__.py doesnt take care of this
if not os.environ.has_key('QUTIP_GRAPHICS'):
    os.environ['QUTIP_GRAPHICS']="YES"

###-platform dependent imports-###
if sys.platform=='darwin':
    from mac import ProgressBar
elif sys.platform=='linux2' and os.environ['QUTIP_GRAPHICS'] == "YES":
	try:
		import pygtk
	except:
		os.environ['QUTIP_GRAPHICS'] == "NO"
	else:
		from linux import ProgressBar

#---------------------------------

class Counter():
    def __init__(self,max,title="Monte-Carlo Trajectories", step_size=1):
        self.max=max
        self.count=0
        self.step=step_size
        self.percent=0.0
        if sys.platform=='darwin' or sys.platform=='linux2' and os.environ['QUTIP_GRAPHICS'] == "YES":
            self.bar = ProgressBar(title="Running "+title)
            self.bar.update(self.percent*100.0, "Completed steps: "+str(self.count)+"/"+str(self.max))
        else:
			self.level=0.1
			print 'Starting Monte-Carlo:'
    def update(self):#update the counter
        self.count+=self.step
        self.percent=self.count/(1.0*self.max)
        if sys.platform=='darwin' or sys.platform=='linux2' and os.environ['QUTIP_GRAPHICS'] == "YES":
            self.bar.update(self.percent*100.0, "Completed steps: "+str(self.count)+"/"+str(self.max))
        else:
            if self.count/float(self.max)>=self.level:
				print str(floor(self.count/float(self.max)*100))+'%  ('+str(self.count)+'/'+str(self.max)+')'
				self.level+=0.1
    
    def finish(self):#pause and close and finish
        time.sleep(0.5)
        if sys.platform=='darwin' or sys.platform=='linux2' and os.environ['QUTIP_GRAPHICS'] == "YES":
            self.bar.finish()
        else:
            print 'All steps completed.'


######---Demo---######
if __name__ == "__main__":
    x=Counter(100)
    def func(x):
        for i in range(100):
            time.sleep(.05)
            x.update()
        time.sleep(.5)
        x.finish()
    
    func(x)

    print 'finished'









           
