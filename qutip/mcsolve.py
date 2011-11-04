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
from scipy.linalg import norm
from Qobj import *
from expect import *
import sys,os,time
from istests import *
from Odeoptions import Odeoptions
import mcdata
import datetime
from multiprocessing import Pool,cpu_count
from varargout import varargout
from types import FunctionType
from tidyup import tidyup
from cyQ.cy_mc_funcs import mc_expect,spmv
from cyQ.ode_rhs import cyq_ode_rhs

def mcsolve(H,psi0,tlist,ntraj,collapse_ops,expect_ops,H_args=None,options=Odeoptions()):
    vout=varargout()
    """
    Monte-Carlo evolution of a state vector |psi> for a given
    Hamiltonian and sets of collapse operators and operators
    for calculating expectation values. Options for solver are 
    given by the Odeoptions class.
    
    .. note:: The number of outputs varies.  See below for details!
    
    Args:
        
        H (Qobj): Hamiltonian.
        
        psi0 (Qobj): Initial state vector.
        
        tlist (list/array): Times at which results are recorded.
        
        ntraj (integer): Number of trajectories to run.
        
        collapse_ops (list/array of Qobj's): Collapse operators.
        
        expect_ops (list/array of Qobj's) Operators for calculating expectation values.
        
        H_args (list/array of Qobj's): Arguments for time-dependent Hamiltonians.
        
        options (Odeoptions): Instance of ODE solver options.
    
    Returns:
        
        Collapse ops  Expectation ops  Num. of outputs  Return value(s)
        ------------  ---------------  ---------------  ---------------
            NO	            NO	              1	         List of state vectors
            
            NO	            YES	              1	         List of expectation values
            
            YES	            NO	              1          List of state vectors for each trajectory
            
            YES	            NO	              2	         List of state vectors for each trajectory 
                                                         + List of collapse times for each trajectory
                                                         
            YES	            NO	              3	         List of state vectors for each trajectory 
                                                         + List of collapse times for each trajectory 
                                                         + List of which operator did collapse for each trajectory
                                                         
            YES	            YES	              1	         List of expectation values for each trajectory
            
            YES	            YES	              2	         List of expectation values for each trajectory 
                                                         + List of collapse times for each trajectory
                                                         
            YES	            YES	              3	         List of expectation values for each trajectory 
                                                         + List of collapse times for each trajectory 
                                                         + List of which operator did collapse for each trajectory
    
    """
    #if Hamiltonian is time-dependent (list style)
    if isinstance(H,(list,ndarray)):
        mcdata.tflag=1
        mcdata.Hfunc=H[0]
        mcdata.Hargs=-1.0j*array([op.data for op in H[1]])
        if len(collapse_ops)>0:
            Hq=0
            for c_op in collapse_ops:
                Hq -= 0.5j * (c_op.dag()*c_op)
            mcdata.Hcoll=-1.0j*Hq.data
    #if Hamiltonian is time-dependent (H_args style)
    elif isinstance(H,FunctionType):
        mcdata.tflag=1
        mcdata.Hfunc=H
        mcdata.Hargs=-1.0j*array([op.data for op in H_args])
        if len(collapse_ops)>0:
            Hq=0
            for c_op in collapse_ops:
                Hq -= 0.5j * (c_op.dag()*c_op)
            mcdata.Hcoll=-1.0j*Hq.data
    #if Hamiltonian is time-independent
    else:
        for c_op in collapse_ops:
            H -= 0.5j * (c_op.dag()*c_op)
        if options.tidy:
            H=tidyup(H,options.atol)
        mcdata.H=-1.0j*H.data 

    mc=MC_class(psi0,tlist,ntraj,collapse_ops,expect_ops,options)
    mc.run()
    if mc.num_collapse==0 and mc.num_expect==0:
        return mc.psi_out
    elif mc.num_collapse==0 and mc.num_expect!=0:
        return mc.expect_out
    elif mc.num_collapse!=0 and mc.num_expect==0:
        if vout==2:
            return mc.psi_out,mc.collapse_times_out
        elif vout==3:
            return mc.psi_out,mc.collapse_times_out,mc.which_op_out
        else:
            return mc.psi_out
    elif mc.num_collapse!=0 and mc.num_expect!=0:
        if vout==2:
            return sum(mc.expect_out,axis=0)/float(ntraj),mc.collapse_times_out
        elif vout==3:
            return sum(mc.expect_out,axis=0)/float(ntraj),mc.collapse_times_out,mc.which_op_out
        else:
            return sum(mc.expect_out,axis=0)/float(ntraj)


#--Monte-Carlo class---
class MC_class():
    """
    Private class for solving Monte-Carlo evolution from mcsolve
    """
    def __init__(self,psi0,tlist,ntraj,collapse_ops,expect_ops,options):
        #-Check for PyObjC on Mac platforms
        self.gui=True
        if sys.platform=='darwin':
            try:
                import Foundation
            except:
                self.gui=False
        ##holds instance of the ProgressBar class
        self.bar=None
        ##holds instance of the Pthread class
        self.thread=None
        ##number of Monte-Carlo trajectories
        self.ntraj=ntraj
        #Number of completed trajectories
        self.count=0
        ##step-size for count attribute
        self.step=1
        ##Percent of trajectories completed
        self.percent=0.0
        ##used in implimenting the command line progress ouput
        self.level=0.1
        ##collection of ODE options from Mcoptions class
        self.options=options
        ##times at which to output state vectors or expectation values
        self.times=tlist
        ##number of time steps in tlist
        self.num_times=len(tlist)
        ##Collapse operators
        self.collapse_ops=collapse_ops
        ##Number of collapse operators
        self.num_collapse=len(collapse_ops)
        ##Operators for calculating expectation values
        self.expect_ops=expect_ops
        ##Number of expectation value operators
        self.num_expect=len(expect_ops)
        ##Matrix representing effective Hamiltonian Heff*1j
        ##Matrix representing initial state vector
        self.psi_in=psi0.full() #need dense matrix as input to ODE solver.
        ##Dimensions of initial state vector
        self.psi_dims=psi0.dims
        ##Shape of initial state vector
        self.psi_shape=psi0.shape
        self.seed=None
        self.st=None #for expected time to completion
        self.cpus=int(os.environ['NUM_THREADS'])
        #FOR EVOLUTION FOR NO COLLAPSE OPERATORS---------------------------------------------
        if self.num_collapse==0:
            if self.num_expect==0:
                ##Output array of state vectors calculated at times in tlist
                self.psi_out=array([Qobj() for k in xrange(self.num_times)])#preallocate array of Qobjs
            elif self.num_expect!=0:#no collpase expectation values
                ##List of output expectation values calculated at times in tlist
                self.expect_out=[]
                ##Array indicating whether expectation operators are Hermitian
                for jj in xrange(self.num_expect):#expectation operators evaluated at initial conditions
                    if self.expect_ops[jj].isherm:
                        self.expect_out.append(zeros(self.num_times))
                    else:
                        self.expect_out.append(zeros(self.num_times,dtype=complex))
        #FOR EVOLUTION WITH COLLAPSE OPERATORS---------------------------------------------
        elif self.num_collapse!=0:
            ##Array of collapse operators A.dag()*A
            self.norm_collapse=array([op.dag()*op for op in self.collapse_ops])
            ##Array of matricies from norm_collpase_data
            self.collapse_ops_data=array([op.data for op in self.collapse_ops])
            #preallocate #ntraj arrays for state vectors, collapse times, and which operator
            self.collapse_times_out=zeros((self.ntraj),dtype=ndarray)
            self.which_op_out=zeros((self.ntraj),dtype=ndarray)
            if self.num_expect==0:# if no expectation operators, preallocate #ntraj arrays for state vectors
                self.psi_out=array([[Qobj() for k in xrange(self.num_times)] for q in xrange(self.ntraj)])#preallocate array of Qobjs
            else: #preallocate array of lists for expectation values
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
        if os.environ['QUTIP_GRAPHICS']=="NO" or os.environ['QUTIP_GUI']=="NONE" or self.gui==False:
            self.percent=self.count/(1.0*self.ntraj)
            if self.count/float(self.ntraj)>=self.level:
                nwt=datetime.datetime.now()
                diff=((nwt.day-self.st.day)*86400+(nwt.hour-self.st.hour)*(60**2)+(nwt.minute-self.st.minute)*60+(nwt.second-self.st.second))*(self.ntraj-self.count)/(1.0*self.count)
                secs=datetime.timedelta(seconds=ceil(diff))
                dd = datetime.datetime(1,1,1) + secs
                time_string="%02d:%02d:%02d:%02d" % (dd.day-1,dd.hour,dd.minute,dd.second)
                print str(floor(self.count/float(self.ntraj)*100))+'%  ('+str(self.count)+'/'+str(self.ntraj)+')'+'  Est. time remaining: '+time_string
                self.level+=0.1
    #########################
    def parallel(self,args,top=None):  
        pl=Pool(processes=self.cpus)
        self.st=datetime.datetime.now()
        for nt in xrange(0,self.ntraj):
            pl.apply_async(mc_alg_evolve,args=(nt,args),callback=top.callback)
        pl.close()
        try:
            pl.join()
        except KeyboardInterrupt:
            print "Cancel all MC threads on keyboard interrupt"
            pl.terminate()
            pl.join()
        return
    def run(self):
        if self.num_collapse==0:
            if self.ntraj!=1:#check if ntraj!=1 which is pointless for no collapse operators
                print 'No collapse operators specified.\nRunning a single trajectory only.\n'
            if self.num_expect==0:# return psi Qobj at each requested time 
                self.psi_out=no_collapse_psi_out(self.options,self.psi_in,self.times,self.num_times,self.psi_dims,self.psi_shape,self.psi_out)
            else:# return expectation values of requested operators
                self.expect_out=no_collapse_expect_out(self.options,self.psi_in,self.times,self.expect_ops,self.num_expect,self.num_times,self.psi_dims,self.psi_shape,self.expect_out)
        elif self.num_collapse!=0:
            self.seed=array([int(ceil(random.rand()*1e4)) for ll in xrange(self.ntraj)])
            args=(self.options,self.psi_in,self.times,self.num_times,self.num_collapse,self.collapse_ops_data,self.norm_collapse,self.num_expect,self.expect_ops,self.seed)
            if os.environ['QUTIP_GRAPHICS']=="NO" or os.environ['QUTIP_GUI']=="NONE" or self.gui==False:
                print 'Starting Monte-Carlo:'
                self.parallel(args,self)
            else:
                from gui.ProgressBar import ProgressBar,Pthread
                if os.environ['QUTIP_GUI']=="PYSIDE":
                    from PySide import QtGui,QtCore
                elif os.environ['QUTIP_GUI']=="PYQT4":
                    from PyQt4 import QtGui,QtCore
                app=QtGui.QApplication.instance()#checks if QApplication already exists (needed for iPython)
                if not app:#create QApplication if it doesnt exist
                    app = QtGui.QApplication(sys.argv)
                thread=Pthread(target=self.parallel,args=args,top=self)
                self.bar=ProgressBar(self,thread,self.ntraj)
                QtCore.QTimer.singleShot(0,self.bar.run)
                self.bar.show()
                self.bar.activateWindow()
                self.bar.raise_()
                app.exec_()
                return
                
            

#RHS of ODE for time-independent systems
def RHS(t,psi):
    return mcdata.H*psi

#RHS of ODE for time-dependent systems with no collapse operators
def RHStd(t,psi):
    return mcdata.Hfunc(t,mcdata.Hargs)*psi

#RHS of ODE for time-dependent systems with collapse operators
def cRHStd(t,psi):
    return (mcdata.Hfunc(t,mcdata.Hargs)+mcdata.Hcoll)*psi


######---return psi at requested times for no collapse operators---######
def no_collapse_psi_out(opt,psi_in,tlist,num_times,psi_dims,psi_shape,psi_out):
    ##Calculates state vectors at times tlist if no collapse AND no expectation values are given.
    #
    if mcdata.tflag==1:
        ODE=ode(RHStd)
    else:
        #ODE=ode(RHS)
        ODE = ode(cyq_ode_rhs)
        ODE.set_f_params(mcdata.H.data, mcdata.H.indices, mcdata.H.indptr)
        
    ODE.set_integrator('zvode',method=opt.method,order=opt.order,atol=opt.atol,rtol=opt.rtol,nsteps=opt.nsteps,first_step=opt.first_step,min_step=opt.min_step,max_step=opt.max_step) #initialize ODE solver for RHS
    ODE.set_initial_value(psi_in,tlist[0]) #set initial conditions
    psi_out[0]=Qobj(psi_in,dims=psi_dims,shape=psi_shape)
    for k in xrange(1,num_times):
        ODE.integrate(tlist[k],step=0) #integrate up to tlist[k]
        if ODE.successful():
            psi_out[k]=Qobj(ODE.y/norm(ODE.y,2),dims=psi_dims,shape=psi_shape)
        else:
            raise ValueError('Error in ODE solver')
    return psi_out
#------------------------------------------------------------------------


######---return expectation values at requested times for no collapse operators---######
def no_collapse_expect_out(opt,psi_in,tlist,expect_ops,num_expect,num_times,psi_dims,psi_shape,expect_out):
    ##Calculates xpect.values at times tlist if no collapse ops. given
    #  
    #------------------------------------
    if mcdata.tflag==1:
        ODE=ode(RHStd)
    else:
        #ODE=ode(RHS)
        ODE = ode(cyq_ode_rhs)
        ODE.set_f_params(mcdata.H.data, mcdata.H.indices, mcdata.H.indptr)

    ODE.set_integrator('zvode',method=opt.method,order=opt.order,atol=opt.atol,rtol=opt.rtol,nsteps=opt.nsteps,first_step=opt.first_step,min_step=opt.min_step,max_step=opt.max_step) #initialize ODE solver for RHS
    ODE.set_initial_value(psi_in,tlist[0]) #set initial conditions
    for jj in xrange(num_expect):
        expect_out[jj][0]=mc_expect(expect_ops[jj],psi_in)
    for k in xrange(1,num_times):
        ODE.integrate(tlist[k],step=0) #integrate up to tlist[k]
        if ODE.successful():
            state=ODE.y/norm(ODE.y)
            for jj in xrange(num_expect):
                expect_out[jj][k]=mc_expect(expect_ops[jj],state)
        else:
            raise ValueError('Error in ODE solver')
    return expect_out #return times and expectiation values
#------------------------------------------------------------------------


#---single-trajectory for monte-carlo---          
def mc_alg_evolve(nt,args):
    """
    Monte-Carlo algorithm returning state-vector or expect. values at times tlist for a single trajectory
    """
    opt,psi_in,tlist,num_times,num_collapse,collapse_ops_data,norm_collapse,num_expect,expect_ops,seeds=args
    if num_expect==0:
        psi_out=array([zeros((len(psi_in),1),dtype=complex) for k in xrange(num_times)])#preallocate real array of Qobjs
        psi_out[0]=psi_in
    else:
        #PRE-GENERATE LIST OF EXPECTATION VALUES
        expect_out=[]
        for i in xrange(num_expect):
            if expect_ops[i].isherm:#preallocate real array of zeros
                expect_out.append(zeros(num_times))
                expect_out[i][0]=mc_expect(expect_ops[i],psi_in)
            else:#preallocate complex array of zeros
                expect_out.append(zeros(num_times,dtype=complex))
                expect_out[i][0]=mc_expect(expect_ops[i],psi_in)    
    #CREATE BLANK LISTS FOR COLLAPSE TIMES AND WHICH OPER
    collapse_times=[] #times at which collapse occurs
    which_oper=[] # which operator did the collapse
    #SEED AND RNG AND GENERATE
    random.seed(seeds[nt])
    rand_vals=random.rand(2)#first rand is collapse norm, second is which operator
    #CREATE ODE OBJECT CORRESPONDING TO RHS
    if mcdata.tflag==1:
        ODE=ode(cRHStd)
    else:
        #ODE=ode(RHS)
        ODE = ode(cyq_ode_rhs)
        ODE.set_f_params(mcdata.H.data, mcdata.H.indices, mcdata.H.indptr)

    ODE.set_integrator('zvode',method=opt.method,order=opt.order,atol=opt.atol,rtol=opt.rtol,nsteps=opt.nsteps,first_step=opt.first_step,min_step=opt.min_step,max_step=opt.max_step) #initialize ODE solver for RHS
    ODE.set_initial_value(psi_in,tlist[0]) #set initial conditions
    #RUN ODE UNTIL EACH TIME IN TLIST
    cinds=arange(num_collapse)
    for k in xrange(1,num_times):
        #last_t=ODE.t;last_y=ODE.y
        #ODE WHILE LOOP FOR INTEGRATE UP TO TIME TLIST[k]
        while ODE.successful() and ODE.t<tlist[k]:
            last_t=ODE.t;last_y=ODE.y
            ODE.integrate(tlist[k],step=1) #integrate up to tlist[k], one step at a time.
            psi_nrm2=norm(ODE.y,2)**2
            if psi_nrm2<=rand_vals[0]:#collpase has occured
                collapse_times.append(ODE.t)
                n_dp=array([mc_expect(op,ODE.y) for op in norm_collapse])
                kk=cumsum(n_dp/sum(n_dp))
                j=cinds[kk>=rand_vals[1]][0]
                which_oper.append(j) #record which operator did collapse
                state=spmv(collapse_ops_data[j].data,collapse_ops_data[j].indices,collapse_ops_data[j].indptr,ODE.y)
                state_nrm=norm(state,2)
                ODE.set_initial_value(state/state_nrm,ODE.t)
                rand_vals=random.rand(2)
            #last_t=ODE.t;last_y=ODE.y
        #-------------------------------------------------------
        ###--after while loop--####
        psi=copy(ODE.y)
        if ODE.t>last_t:
            psi=(psi-last_y)/(ODE.t-last_t)*(tlist[k]-last_t)+last_y
        psi_nrm=norm(psi,2)
        if num_expect==0:
            psi_out[k] = psi/psi_nrm
        else:
            epsi=psi/psi_nrm
            for jj in xrange(num_expect):
                expect_out[jj][k]=mc_expect(expect_ops[jj],epsi)
    #RETURN VALUES
    if num_expect==0:
        return nt,psi_out,array(collapse_times),array(which_oper)
    else:
        return nt,expect_out,array(collapse_times),array(which_oper)
#------------------------------------------------------------------------------------------












