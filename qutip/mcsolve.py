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
from scipy import *
from scipy.integrate import *
import scipy.linalg as la
from Qobj import *
from expect import *
import sys,os,time
from istests import *
from Mcoptions import Mcoptions


def mcsolve(H,psi0,tlist,ntraj,collapse_ops,expect_ops,options=Mcoptions()):

    Heff = H
    for c_op in collapse_ops:
        Heff -= 0.5j * c_op.dag() * c_op 

    mc=MC_class(Heff,psi0,tlist,ntraj,collapse_ops,expect_ops,options)
    mc.run()
    if mc.num_collapse==0 and mc.num_expect==0:
        return mc.psi_out
    elif mc.num_collapse==0 and mc.num_expect!=0:
        if options.output_trajectories:
            return mc.expect_out
        else:
            return sum(mc.expect_out,axis=0)/ntraj
    elif mc.num_collapse!=0 and mc.num_expect==0:
        return mc.psi_out
        #return mc.psi_out, mc.collapse_times_out
    elif mc.num_collapse!=0 and mc.num_expect!=0:
        if options.output_trajectories:
            return mc.expect_out
        else:
            return sum(mc.expect_out,axis=0)/ntraj
       


######---Monte-Carlo class---######
class MC_class():
    def __init__(self,Heff,psi0,tlist,ntraj,collapse_ops,expect_ops,options):
        self.bar=None
        self.thread=None
        self.max=ntraj
        self.count=0
        self.step=1
        self.percent=0.0
        self.level=0.1
        self.options=options
        self.times=tlist
        self.ntraj=ntraj
        self.num_times=len(tlist)
        self.collapse_ops=collapse_ops
        self.num_collapse=len(collapse_ops)
        self.expect_ops=expect_ops
        self.num_expect=len(expect_ops)
        self.Hdata=-1.0j*Heff.data# extract matrix and multiply Heff by -1j so user doesn't have to.
        self.psi_in=psi0.full() #need dense matrix as input to ODE solver.
        self.psi_dims=psi0.dims
        self.psi_shape=psi0.shape

        if self.num_collapse==0:
            if self.num_expect==0:
                self.psi_out=array([Qobj() for k in xrange(self.num_times)])#preallocate array of Qobjs
            elif self.num_expect!=0:#no collpase expectation values
                self.expect_out=[]
                self.isher=isherm(self.expect_ops)#checks if expectation operators are hermitian
                for jj in xrange(self.num_expect):#expectation operators evaluated at initial conditions
                    if self.isher[jj]==1:
                        self.expect_out.append(zeros(self.num_times))
                    else:
                        self.expect_out.append(zeros(self.num_times,dtype=complex))
        elif self.num_collapse!=0:
            #extract matricies from collapse operators
            self.norm_collapse_data=array([(op.dag()*op).data for op in self.collapse_ops])
            self.collapse_ops_data=array([op.data for op in self.collapse_ops])
            #preallocate #ntraj arrays for state vectors, collapse times, and which operator
            self.collapse_times_out=zeros((self.ntraj),dtype=ndarray)
            self.which_op_out=zeros((self.ntraj),dtype=ndarray)
            if self.num_expect==0:# if no expectation operators, preallocate #ntraj arrays for state vectors
                self.isher = None
                self.psi_out=array([Qobj() for k in xrange(self.num_times)])#preallocate array of Qobjs
            else: #preallocate array of lists for expectation values
                self.isher=[ops.isherm for ops in self.expect_ops]
                self.expect_out=[[] for x in xrange(self.ntraj)]
    #------------------------------------------------------------------------------------
    def callback(self,results):
        r=results[0]
        if self.num_expect==0:#output state-vector
            self.psi_out[r]=array([Qobj(psi,dims=self.psi_dims,shape=self.psi_shape) for psi in results[1]])
        else:#output expectation values
            self.expect_out[r]=results[1]
        self.collapse_times_out[r]=results[2]
        self.which_op_out[r]=results[3]
        self.count+=self.step
        if os.environ['QUTIP_GUI']=="NONE":
            self.percent=self.count/(1.0*self.max)
            if self.count/float(self.max)>=self.level:
                print str(floor(self.count/float(self.max)*100))+'%  ('+str(self.count)+'/'+str(self.max)+')'
                self.level+=0.1
    #########################
    def parallel(self,args,top=None):
        from multiprocessing import Pool,cpu_count
        pl=Pool(processes=cpu_count())
        for nt in xrange(0,self.ntraj):
            pl.apply_async(mc_alg_evolve,args=(nt,args),callback=top.callback)
        pl.close()
        pl.join()
        return
    def run(self):
        if self.num_collapse==0:
            if self.ntraj!=1:#check if ntraj!=1 which is pointless for no collapse operators
                print 'No collapse operators specified.\nRunning a single trajectory only.\n'
            if self.num_expect==0:# return psi Qobj at each requested time 
                self.psi_out=no_collapse_psi_out(self.options,self.Hdata,self.psi_in,self.times,self.num_times,self.psi_dims,self.psi_shape,self.psi_out)
            else:# return expectation values of requested operators
                self.expect_out=no_collapse_expect_out(self.options,self.Hdata,self.psi_in,self.times,self.expect_ops,self.num_expect,self.num_times,self.psi_dims,self.psi_shape,self.expect_out,self.isher)
        elif self.num_collapse!=0:
            args=(self.options,self.Hdata,self.psi_in,self.times,self.num_times,self.num_collapse,self.collapse_ops_data,self.norm_collapse_data,self.num_expect,self.expect_ops,self.isher)
            if os.environ['QUTIP_GUI']=="NONE":
                print 'Starting Monte-Carlo:'
                self.parallel(args,self)
            else:
                from gui import ProgressBar,Pthread
                if os.environ['QUTIP_GUI']=="PYSIDE":
                    from PySide import QtGui,QtCore
                elif os.environ['QUTIP_GUI']=="PYQT4":
                    from PyQt4 import QtGui,QtCore
                app=QtGui.QApplication.instance()#checks if QApplication already exists (needed for iPython)
                if not app:#create QApplication if it doesnt exist
                    app = QtGui.QApplication(sys.argv)
                thread=Pthread(target=self.parallel,args=args,top=self)
                bar=ProgressBar(self,thread,self.max)
                QtCore.QTimer.singleShot(0,bar.run)
                bar.show()
                bar.raise_()
                app.exec_()
                return
                
            



######---return psi at requested times for no collapse operators---######
def no_collapse_psi_out(opt,Hdata,psi_in,tlist,num_times,psi_dims,psi_shape,psi_out):
    def RHS(t,psi):#define RHS of ODE
            return Hdata*psi #cannot use dot(a,b) since mat is matrix and not array.
    ODE=ode(RHS)
    ODE.set_integrator('zvode',method=opt.method,order=opt.order,atol=opt.atol,rtol=opt.rtol,nsteps=opt.nsteps,first_step=opt.first_step,min_step=opt.min_step,max_step=opt.max_step) #initialize ODE solver for RHS
    ODE.set_initial_value(psi_in,tlist[0]) #set initial conditions
    psi_out[0]=Qobj(psi_in,dims=psi_dims,shape=psi_shape)
    for k in xrange(1,num_times):
        ODE.integrate(tlist[k],step=0) #integrate up to tlist[k]
        if ODE.successful():
            psi_out[k]=Qobj(ODE.y/la.norm(ODE.y),dims=psi_dims,shape=psi_shape)
        else:
            raise ValueError('Error in ODE solver')
    return psi_out
#------------------------------------------------------------------------


######---return expectation values at requested times for no collapse operators---######
def no_collapse_expect_out(opt,Hdata,psi_in,tlist,expect_ops,num_expect,num_times,psi_dims,psi_shape,expect_out,isher):
    ######---Define RHS of ODE---##############
    def RHS(t,psi):
            return Hdata*psi #cannot use dot(a,b) since mat is matrix and not array.
    ######-------------------------------------
    ODE=ode(RHS)
    ODE.set_integrator('zvode',method=opt.method,order=opt.order,atol=opt.atol,rtol=opt.rtol,nsteps=opt.nsteps,first_step=opt.first_step,min_step=opt.min_step,max_step=opt.max_step) #initialize ODE solver for RHS
    ODE.set_initial_value(psi_in,tlist[0]) #set initial conditions
    for jj in xrange(num_expect):
        expect_out[jj][0]=mc_expect(expect_ops[jj],psi_in,isher[jj])
    for k in xrange(1,num_times):
        ODE.integrate(tlist[k],step=0) #integrate up to tlist[k]
        if ODE.successful():
            state=ODE.y/la.norm(ODE.y)
            for jj in xrange(num_expect):
                expect_out[jj][k]=mc_expect(expect_ops[jj],state,isher[jj])
        else:
            raise ValueError('Error in ODE solver')
    return expect_out #return times and expectiation values
#------------------------------------------------------------------------


######---single-trajectory for monte-carlo---###########           
def mc_alg_evolve(nt,args):
    """
    Monte-Carlo algorithm returning state-vector at times tlist[k] for a single trajectory
    """
    opt,Hdata,psi_in,tlist,num_times,num_collapse,collapse_ops_data,norm_collapse_data,num_expect,expect_ops,isher=args
    def RHS(t,psi):
            return Hdata*psi #cannot use dot(a,b) since mat is matrix and not array.
    if num_expect==0:
        psi_out=array([zeros((len(psi_in),1),dtype=complex) for k in xrange(num_times)])#preallocate real array of Qobjs
        psi_out[0]=psi_in
    else:
        expect_out=[]
        for i in isher:
            if i==1:#preallocate real array of zeros
                expect_out.append(zeros(num_times))
            else:#preallocate complex array of zeros
                expect_out.append(zeros(num_times,dtype=complex))    
        for jj in xrange(num_expect):
            expect_out[jj][0]=mc_expect(expect_ops[jj],psi_in,isher[jj])
    collapse_times=[] #times at which collapse occurs
    which_oper=[] # which operator did the collapse
    random.seed()
    rand_vals=random.random(2)#first rand is collapse norm, second is which operator
    ODE=ode(RHS)
    ODE.set_integrator('zvode',method=opt.method,order=opt.order,atol=opt.atol,rtol=opt.rtol,nsteps=opt.nsteps,first_step=opt.first_step,min_step=opt.min_step,max_step=opt.max_step) #initialize ODE solver for RHS
    ODE.set_initial_value(psi_in,tlist[0]) #set initial conditions
    for k in xrange(1,num_times):
        last_t=ODE.t;last_y=ODE.y
        step_flag=1
        while ODE.successful() and ODE.t<tlist[k]:
            ODE.integrate(tlist[k],step=step_flag) #integrate up to tlist[k], one step at a time.
            if ODE.t>tlist[k]:
                ODE.set_integrator('zvode',first_step=(tlist[k]-last_t),method=opt.method,order=opt.order,atol=opt.atol,rtol=opt.rtol,nsteps=opt.nsteps,min_step=opt.min_step,max_step=opt.max_step)
                ODE.set_initial_value(last_y,last_t)
                step_flag=0
            else:
                psi_nrm=la.norm(ODE.y)
                if psi_nrm<0.0:#safety check for norm<0
                    psi_nrm,ODE=norm_safety(ODE,tlist,psi_nrm,last_y,last_t)#find non-zero psi norm
                if psi_nrm<=rand_vals[0]:#collpase has occured
                    collapse_times.append(ODE.t)
                    m=0.0
                    n_dp=array([real(dot(ODE.y.conj().T,op*ODE.y)[0,0]) for op in norm_collapse_data])
                    dp=sum(n_dp)
                    for j in xrange(num_collapse):
                        m+=n_dp[j]/dp
                        if m>=rand_vals[1]:
                            which_oper.append(j) #record which operator did collapse
                            state=collapse_ops_data[j]*ODE.y
                            psi_nrm=la.norm(state)
                            state=state/psi_nrm
                            ODE.y=state
                            random.seed()
                            rand_vals=random.random(2)
                            #last_y=ODE.y;last_t=ODE.t
                            break #breaks out of for-loop
                last_t=ODE.t;last_y=ODE.y
        ###--after while loop--####
        psi_nrm=la.norm(ODE.y)
        if num_expect==0:
            psi_out[k] = ODE.y/psi_nrm
        else:
            state=ODE.y/psi_nrm
            for jj in xrange(num_expect):
                expect_out[jj][k]=mc_expect(expect_ops[jj],state,isher[jj])
    if num_expect==0:
        return nt,psi_out,array(collapse_times),array(which_oper)
    else:
        return nt,expect_out,array(collapse_times),array(which_oper)
#------------------------------------------------------------------------------------------



######---if psi norm is less than zero---########### 
def norm_safety(ODE,tlist,psi_nrm,last_y,last_t):
    print 'wavefunction norm below zero, reducing step size.'
    ntrys=1
    while ODE.successful() and la.norm(ODE.y)<0 and ntrys<=10:#reduce step-size by half and integrate again until norm>0 or more than 10 attempts
        ODE.set_integrator('zvode',method='adams',nsteps=500,atol=1e-6,first_step=(ODE.t-last_t)/2.0)
        ODE.set_initial_value(last_y,last_t)
        ODE.integrate(tlist[k],step=1)
        ntrys+=1
    psi_nrm=la.norm(ODE.y)
    if psi_nrm<0:# if norm still <0 return error
        raise ValueError('State vector norm<0 after reducing ODE step-size 10 times.')
    ODE.set_integrator('zvode',method='adams',nsteps=500,atol=1e-6)
    return psi_nrm,ODE
#------------------------------------------------------------------------


def mc_expect(oper,state,isherm):
    if isherm==1:
        return real(dot(conj(state).T,oper.data*state))
    else:
        return complex(dot(conj(state).T,oper.data*state))











