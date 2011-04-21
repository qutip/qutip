from scipy import *
from scipy.integrate import *
import scipy.sparse as sp
#from pylab import *
import scipy.linalg as la
import time,sys
from multiprocessing import Pool,cpu_count
from qutip.Counter import *
from qutip import *
def RHS(t,vec):
    mat=sp.csr_matrix([[.2,sqrt(t)],[.01j,-1]])
    ret=mat*vec #cannot use dot(a,b) since mat is mtrix and not array
    return ret


y0=array([1./sqrt(2),1./sqrt(2)]) #to column vec
t0=0
t1=20

ODE=ode(RHS).set_integrator('zvode',method='adams',nsteps=1000,atol=1e-12) #initialize ODE solver for RHS
ODE.set_initial_value(y0,t0) #set initial conditions

y=array([qobj() for x in range(10)])
#bar=Counter(10)
num=0
num_max=10
def cb(r):
    global num,num_max
    y[num]=qobj(r)
    num=num+1
    print str(100.0*num/num_max)+' %'
    #bar.update()


def odefunc(x):
    num_steps=0.
    ODE.set_initial_value(y0,t0) #set initial conditions
    while ODE.successful() and ODE.t<t1:
        ODE.integrate(t1,step=1)
        num_steps+=1.
        if la.norm(ODE.y)>=1.5:
            ODE.set_initial_value(y0,ODE.t)
    return num_steps

po = Pool(processes=cpu_count())
start=time.time()
if sys.platform=='darwin':
    for i in xrange(10):
        po.apply_async(odefunc,(i,),callback=cb)
po.close()
po.join()
#bar.finish()
finish=time.time()
print 'ellapsed time = ',finish-start
print y
