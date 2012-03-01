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
# Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson
#
###########################################################################
import sys,os,time,numpy,datetime
from scipy import *
from scipy.integrate import *
from scipy.linalg import norm
from qutip.Qobj import *
from qutip.expect import *
from qutip.istests import *
from qutip.Odeoptions import Odeoptions
import qutip.odeconfig as odeconfig
from multiprocessing import Pool,cpu_count
from types import FunctionType
from qutip.tidyup import tidyup
from qutip.cyQ.cy_mc_funcs import mc_expect,spmv,cy_mc_no_time
from qutip.cyQ.ode_rhs import cyq_ode_rhs
from qutip.cyQ.codegen import Codegen
from qutip.rhs_generate import rhs_generate
from Mcdata import Mcdata

def mcsolve(H,psi0,tlist,ntraj,c_ops,e_ops,args={},options=Odeoptions()):
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
        
        c_ops (list/array of Qobj's): Collapse operators.
        
        e_ops (list/array of Qobj's) Operators for calculating expectation values.
        
        args (list/array of Qobj's): Arguments for time-dependent Hamiltonians.
        
        options (Odeoptions): Instance of ODE solver options.
    
    Returns:
        
        Mcdata object storing all results from simulation.
        
    """
    #reset odeconfig collapse and time-dependence flags to default values
    odeconfig.cflag=0
    odeconfig.tflag=0
    #if Hamiltonian is time-dependent (list form) & cythonize RHS
    if isinstance(H,list):
        if len(H)!=2:
            raise TypeError('Time-dependent Hamiltonian list must have two terms.')
        if (not isinstance(H[0],(list,ndarray))) or (len(H[0])<=1):
            raise TypeError('Time-dependent Hamiltonians must be a list with two or more terms')
        if (not isinstance(H[1],(list,ndarray))) or (len(H[1])!=(len(H[0])-1)):
            raise TypeError('Time-dependent coefficients must be list with length N-1 where N is the number of Hamiltonian terms.')
        odeconfig.tflag=1
        if options.rhs_reuse==True and odeconfig.tdfunc==None:
            print "No previous time-dependent RHS found."
            print "Generating one for you..."
            rhs_generate(H,args)
        lenh=len(H[0])
        if options.tidy:
            H[0]=[tidyup(H[0][k]) for k in range(lenh)]
        if len(c_ops)>0:
            odeconfig.cflag=1
            for c_op in c_ops:
                H[0][0]-=0.5j*(c_op.dag()*c_op)
        #create data arrays for time-dependent RHS function
        odeconfig.Hdata=[-1.0j*H[0][k].data.data for k in range(lenh)]
        odeconfig.Hinds=[H[0][k].data.indices for k in range(lenh)]
        odeconfig.Hptrs=[H[0][k].data.indptr for k in range(lenh)]
        #setup ode args string
        odeconfig.string=""
        for k in range(lenh):
            odeconfig.string+="odeconfig.Hdata["+str(k)+"],odeconfig.Hinds["+str(k)+"],odeconfig.Hptrs["+str(k)+"],"
        if len(args)>0:
            td_consts=args.items()
            for elem in td_consts:
                odeconfig.string+=str(elem[1])
                if elem!=td_consts[-1]:
                    odeconfig.string+=(",")
        #run code generator
        if not options.rhs_reuse:
            name="rhs"+str(odeconfig.cgen_num)
            odeconfig.tdname=name
            cgen=Codegen(lenh,H[1],args)
            cgen.generate(name+".pyx")
    # NON-CYTHON TIME-DEPENDENT CODE
    elif isinstance(H,FunctionType):
        odeconfig.tflag=2
        odeconfig.Hfunc=H
        odeconfig.Hargs=-1.0j*array([op.data for op in args])
        if len(c_ops)>0:
            odeconfig.cflag=1
            Hq=0
            for c_op in c_ops:
                Hq -= 0.5j * (c_op.dag()*c_op)
            Hq=tidyup(Hq,options.atol)
            odeconfig.Hcoll=-1.0j*Hq.data
    #if Hamiltonian is time-independent
    else:
        if len(c_ops)>0: odeconfig.cflag=1 #collapse operator flag
        for c_op in c_ops:
            H -= 0.5j * (c_op.dag()*c_op)
        if options.tidy:
            H=tidyup(H,options.atol)
        odeconfig.Hdata=-1.0j*H.data.data
        odeconfig.Hinds=H.data.indices
        odeconfig.Hptrs=H.data.indptr


    mc=MC_class(psi0,tlist,ntraj,c_ops,e_ops,options)
    mc.run()
    #AFTER MCSOLVER IS DONE --------------------------------------
    if odeconfig.tflag==1 and (not options.rhs_reuse):
        os.remove(odeconfig.tdname+".pyx")
    output=Mcdata()
    
    #if any(mc.psi_out) and odeconfig.cflag==1 and options.mc_avg==True:#averaging if multiple trajectories
        #output.states=mean(mc.psi_out,axis=0)
    #else:
    output.states=mc.psi_out
    if any(mc.expect_out) and odeconfig.cflag==1 and options.mc_avg==True:#averaging if multiple trajectories
        output.expect=mean(mc.expect_out,axis=0)
    else:#no averaging for single trajectory or if mc_avg flag (Odeoptions) is off
        output.expect=mc.expect_out
    
    output.times=mc.times
    output.num_expect=mc.num_expect
    output.num_collapse=mc.num_collapse
    output.ntraj=mc.ntraj
    output.collapse_times=mc.collapse_times_out
    output.collapse_which=mc.which_op_out
    return output


#--Monte-Carlo class---
class MC_class():
    """
    Private class for solving Monte-Carlo evolution from mcsolve
    """
    def __init__(self,psi0,tlist,ntraj,c_ops,e_ops,options):
        ##collection of ODE options from Mcoptions class
        self.options=options
        #-Check for PyObjC on Mac platforms
        if sys.platform=='darwin':
            try:
                import Foundation
            except:
                self.options.gui=False
        #check if running in iPython and using Cython compiling (then no GUI to work around error)
        if self.options.gui and odeconfig.tflag==1:
            try:
                __IPYTHON__
            except:
                pass
            else:
                self.options.gui=False    
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
        ##times at which to output state vectors or expectation values
        self.times=tlist
        ##number of time steps in tlist
        self.num_times=len(tlist)
        ##Collapse operators
        self.c_ops=c_ops
        ##Number of collapse operators
        self.num_collapse=len(c_ops)
        ##Operators for calculating expectation values
        self.e_ops=e_ops
        ##Number of expectation value operators
        self.num_expect=len(e_ops)
        ##Matrix representing initial state vector
        self.psi_in=psi0.full() #need dense matrix as input to ODE solver.
        ##Dimensions of initial state vector
        self.psi_dims=psi0.dims
        ##Shape of initial state vector
        self.psi_shape=psi0.shape
        self.seed=None
        self.st=None #for expected time to completion
        self.cpus=options.num_cpus
        #set output variables, even if they are not used to simplify output code.
        self.psi_out=None
        self.expect_out=None
        self.collapse_times_out=None
        self.which_op_out=None
        #FOR EVOLUTION FOR NO COLLAPSE OPERATORS---------------------------------------------
        if self.num_collapse==0:
            if self.num_expect==0:
                ##Output array of state vectors calculated at times in tlist
                self.psi_out=array([Qobj()]*self.num_times)#preallocate array of Qobjs
            elif self.num_expect!=0:#no collpase expectation values
                ##List of output expectation values calculated at times in tlist
                self.expect_out=[]
                for i in xrange(self.num_expect):
                    if self.e_ops[i].isherm:#preallocate real array of zeros
                        self.expect_out.append(zeros(self.num_times))
                        self.expect_out[i][0]=mc_expect(self.e_ops[i].data.data,self.e_ops[i].data.indices,self.e_ops[i].data.indptr,self.e_ops[i].isherm,self.psi_in)
                    else:#preallocate complex array of zeros
                        self.expect_out.append(zeros(self.num_times,dtype=complex))
                        self.expect_out[i][0]=mc_expect(self.e_ops[i].data.data,self.e_ops[i].data.indices,self.e_ops[i].data.indptr,self.e_ops[i].isherm,self.psi_in)
        #FOR EVOLUTION WITH COLLAPSE OPERATORS---------------------------------------------
        elif self.num_collapse!=0:
            ##Array of collapse operators A.dag()*A
            self.norm_collapse=array([(op.dag()*op).data for op in self.c_ops])
            ##Array of matricies from norm_collpase_data
            self.c_ops_data=array([op.data for op in self.c_ops])
            #preallocate #ntraj arrays for state vectors, collapse times, and which operator
            self.collapse_times_out=zeros((self.ntraj),dtype=ndarray)
            self.which_op_out=zeros((self.ntraj),dtype=ndarray)
            if self.num_expect==0:# if no expectation operators, preallocate #ntraj arrays for state vectors
                self.psi_out=array([zeros((self.num_times),dtype=object) for q in xrange(self.ntraj)])#preallocate array of Qobjs
            else: #preallocate array of lists for expectation values
                self.expect_out=[[] for x in xrange(self.ntraj)]
    #------------------------------------------------------------------------------------
    def callback(self,results):
        r=results[0]
        if self.num_expect==0:#output state-vector
            self.psi_out[r]=results[1]
        else:#output expectation values
            self.expect_out[r]=results[1]
        self.collapse_times_out[r]=results[2]
        self.which_op_out[r]=results[3]
        self.count+=self.step
        if not self.options.gui: #do not use GUI
            self.percent=self.count/(1.0*self.ntraj)
            if self.count/float(self.ntraj)>=self.level:
                #calls function to determine simulation time remaining
                self.level=_time_remaining(self.st,self.ntraj,self.count,self.level)
    #########################
    def parallel(self,args,top=None):  
        self.st=datetime.datetime.now() #set simulation starting time
        if sys.platform[0:3]!="win":
            pl=Pool(processes=self.cpus)
            [pl.apply_async(mc_alg_evolve,args=(nt,args),callback=top.callback) for nt in xrange(0,self.ntraj)]
            pl.close()
            try:
                pl.join()
            except KeyboardInterrupt:
                print "Cancel all MC threads on keyboard interrupt"
                pl.terminate()
                pl.join()
            return
        else: # Code for running on Windows (single-cpu only)
            print "Using Windows: Multiprocessing NOT available."
            for nt in xrange(self.ntraj):
                par_return=mc_alg_evolve(nt,args)
                if self.num_expect==0:
                    self.psi_out[nt]=array([Qobj(psi,dims=self.psi_dims,shape=self.psi_shape) for psi in par_return[k][0]])
                else:
                    self.expect_out[nt]=par_return[1]
                    self.collapse_times_out[nt]=par_return[2]
                    self.which_op_out[nt]=par_return[3]
                self.count+=self.step
                self.percent=self.count/(1.0*self.ntraj)
                if self.count/float(self.ntraj)>=self.level:
                    #calls function to determine simulation time remaining
                    self.level=_time_remaining(self.st,self.ntraj,self.count,self.level)
            return
    def run(self):
        if odeconfig.tflag==1: #compile time-depdendent RHS code
            if not self.options.rhs_reuse:
                print "Compiling '"+odeconfig.tdname+".pyx' ..."
                os.environ['CFLAGS'] = '-w'
                import pyximport
                pyximport.install(setup_args={'include_dirs':[numpy.get_include()]})
                code = compile('from '+odeconfig.tdname+' import cyq_td_ode_rhs', '<string>', 'exec')
                exec(code)
                print 'Done.'
                odeconfig.tdfunc=cyq_td_ode_rhs
        if self.num_collapse==0:
            if self.ntraj!=1:#check if ntraj!=1 which is pointless for no collapse operators
                self.ntraj=1
                print('No collapse operators specified.\nRunning a single trajectory only.\n')
            if self.num_expect==0:# return psi Qobj at each requested time 
                self.psi_out=no_collapse_psi_out(self.options,self.psi_in,self.times,self.num_times,self.psi_dims,self.psi_shape,self.psi_out)
            else:# return expectation values of requested operators
                self.expect_out=no_collapse_expect_out(self.options,self.psi_in,self.times,self.e_ops,self.num_expect,self.num_times,self.psi_dims,self.psi_shape,self.expect_out)
        elif self.num_collapse!=0:
            self.seed=array([int(ceil(random.rand()*1e4)) for ll in xrange(self.ntraj)])
            if self.num_expect==0:
                mc_alg_out=zeros((self.num_times),dtype=ndarray)
                mc_alg_out[0]=self.psi_in
            else:
                #PRE-GENERATE LIST OF EXPECTATION VALUES
                mc_alg_out=[]
                for i in xrange(self.num_expect):
                    if self.e_ops[i].isherm:#preallocate real array of zeros
                        mc_alg_out.append(zeros(self.num_times))
                        mc_alg_out[i][0]=mc_expect(self.e_ops[i].data.data,self.e_ops[i].data.indices,self.e_ops[i].data.indptr,self.e_ops[i].isherm,self.psi_in)
                    else:#preallocate complex array of zeros
                        mc_alg_out.append(zeros(self.num_times,dtype=complex))
                        mc_alg_out[i][0]=mc_expect(self.e_ops[i].data.data,self.e_ops[i].data.indices,self.e_ops[i].data.indptr,self.e_ops[i].isherm,self.psi_in)
            
            #set arguments for input to monte-carlo
            args=(self.options,self.psi_in,self.psi_dims,self.psi_shape,mc_alg_out,self.times,self.num_times,self.c_ops_data,self.norm_collapse,self.e_ops,self.seed)
            if (not self.options.gui) or sys.platform[0:3]=="win":
                print('Starting Monte-Carlo:')
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
                


#----------------------------------------------------
#
# CODES FOR PYTHON BASED TIME-DEPENDENT HAMILTONIANS
#
#----------------------------------------------------

#RHS of ODE for time-dependent systems with no collapse operators
def RHStd(t,psi):
    return odeconfig.Hfunc(t,odeconfig.Hargs)*psi

#RHS of ODE for time-dependent systems with collapse operators
def cRHStd(t,psi):
    return (odeconfig.Hfunc(t,odeconfig.Hargs)+odeconfig.Hcoll)*psi





######---return psi at requested times for no collapse operators---######
def no_collapse_psi_out(opt,psi_in,tlist,num_times,psi_dims,psi_shape,psi_out):
    ##Calculates state vectors at times tlist if no collapse AND no expectation values are given.
    #
    if odeconfig.tflag==1:
        ODE=ode(odeconfig.tdfunc)
        code = compile('ODE.set_f_params('+odeconfig.string+')', '<string>', 'exec')
        exec(code)
    elif odeconfig.tflag==2:
        ODE=ode(RHStd)
    else:
        #ODE=ode(RHS)
        ODE = ode(cyq_ode_rhs)
        ODE.set_f_params(odeconfig.Hdata, odeconfig.Hinds, odeconfig.Hptrs)
        
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
def no_collapse_expect_out(opt,psi_in,tlist,e_ops,num_expect,num_times,psi_dims,psi_shape,expect_out):
    ##Calculates xpect.values at times tlist if no collapse ops. given
    #  
    #------------------------------------
    if odeconfig.tflag==1:
        ODE=ode(odeconfig.tdfunc)
        code = compile('ODE.set_f_params('+odeconfig.string+')', '<string>', 'exec')
        exec(code)
    elif odeconfig.tflag==2:
        ODE=ode(RHStd)
    else:
        ODE = ode(cyq_ode_rhs)
        ODE.set_f_params(odeconfig.Hdata, odeconfig.Hinds, odeconfig.Hptrs)
    ODE.set_integrator('zvode',method=opt.method,order=opt.order,atol=opt.atol,rtol=opt.rtol,nsteps=opt.nsteps,first_step=opt.first_step,min_step=opt.min_step,max_step=opt.max_step) #initialize ODE solver for RHS
    ODE.set_initial_value(psi_in,tlist[0]) #set initial conditions
    for jj in xrange(num_expect):
        expect_out[jj][0]=mc_expect(e_ops[jj],psi_in)
    for k in xrange(1,num_times):
        ODE.integrate(tlist[k],step=0) #integrate up to tlist[k]
        if ODE.successful():
            state=ODE.y/norm(ODE.y)
            for jj in xrange(num_expect):
                expect_out[jj][k]=mc_expect(e_ops[jj],state)
        else:
            raise ValueError('Error in ODE solver')
    return expect_out #return times and expectiation values
#------------------------------------------------------------------------


#---single-trajectory for monte-carlo---          
def mc_alg_evolve(nt,args):
    """
    Monte-Carlo algorithm returning state-vector or expect. values at times tlist for a single trajectory
    """
    opt,psi_in,psi_dims,psi_shape,mc_alg_out,tlist,num_times,c_ops_data,norm_collapse_data,e_ops,seeds=args
    num_expect=len(e_ops)
    num_collapse=len(c_ops_data)
    collapse_times=[] #times at which collapse occurs
    which_oper=[] # which operator did the collapse
    
    #SEED AND RNG AND GENERATE
    random.seed(seeds[nt])
    rand_vals=random.rand(2)#first rand is collapse norm, second is which operator
    
    #CREATE ODE OBJECT CORRESPONDING TO RHS
    if odeconfig.tflag==1:
        ODE=ode(odeconfig.tdfunc)
        code = compile('ODE.set_f_params('+odeconfig.string+')', '<string>', 'exec')
        exec(code)
    elif odeconfig.tflag==2:
        ODE=ode(RHStd)
    else:
        ODE = ode(cy_mc_no_time)
        ODE.set_f_params(odeconfig.Hdata, odeconfig.Hinds, odeconfig.Hptrs)
    ODE.set_integrator('zvode',method=opt.method,order=opt.order,atol=opt.atol,rtol=opt.rtol,nsteps=opt.nsteps,first_step=opt.first_step,min_step=opt.min_step,max_step=opt.max_step) #initialize ODE solver for RHS
    ODE.set_initial_value(psi_in,tlist[0]) #set initial conditions
    
    #RUN ODE UNTIL EACH TIME IN TLIST
    cinds=arange(num_collapse)
    for k in xrange(1,num_times):
        #ODE WHILE LOOP FOR INTEGRATE UP TO TIME TLIST[k]
        while ODE.successful() and ODE.t<tlist[k]:
            last_t=ODE.t;last_y=ODE.y
            ODE.integrate(tlist[k],step=1) #integrate up to tlist[k], one step at a time.
            psi_nrm2=norm(ODE.y,2)**2
            if psi_nrm2<=rand_vals[0]:#collpase has occured
                collapse_times.append(ODE.t)
                n_dp=array([mc_expect(op.data,op.indices,op.indptr,1,ODE.y) for op in norm_collapse_data])
                kk=cumsum(n_dp/sum(n_dp))
                j=cinds[kk>=rand_vals[1]][0]
                which_oper.append(j) #record which operator did collapse
                state=spmv(c_ops_data[j].data,c_ops_data[j].indices,c_ops_data[j].indptr,ODE.y)
                state_nrm=norm(state,2)
                ODE.set_initial_value(state/state_nrm,ODE.t)
                rand_vals=random.rand(2)
        #-------------------------------------------------------
        ###--after while loop--####
        psi=copy(ODE.y)
        if ODE.t>last_t:
            psi=(psi-last_y)/(ODE.t-last_t)*(tlist[k]-last_t)+last_y
        psi_nrm=norm(psi,2)
        if num_expect==0:
            mc_alg_out[k]=psi/psi_nrm
        else:
            epsi=psi/psi_nrm
            for jj in xrange(num_expect):
                mc_alg_out[jj][k]=mc_expect(e_ops[jj].data.data,e_ops[jj].data.indices,e_ops[jj].data.indptr,e_ops[jj].isherm,epsi)
    #RETURN VALUES
    if num_expect==0:
        mc_alg_out=array([Qobj(k,psi_dims,psi_shape,'ket') for k in mc_alg_out])
        return nt,mc_alg_out,array(collapse_times),array(which_oper)
    else:
        return nt,mc_alg_out,array(collapse_times),array(which_oper)
#------------------------------------------------------------------------------------------




def _time_remaining(st,ntraj,count,level):
    """
    Private function that determines, and prints, how much simulation time is remaining.
    """
    nwt=datetime.datetime.now()
    diff=((nwt.day-st.day)*86400+(nwt.hour-st.hour)*(60**2)+(nwt.minute-st.minute)*60+(nwt.second-st.second))*(ntraj-count)/(1.0*count)
    secs=datetime.timedelta(seconds=ceil(diff))
    dd = datetime.datetime(1,1,1) + secs
    time_string="%02d:%02d:%02d:%02d" % (dd.day-1,dd.hour,dd.minute,dd.second)
    print str(floor(count/float(ntraj)*100))+'%  ('+str(count)+'/'+str(ntraj)+')'+'  Est. time remaining: '+time_string
    level+=0.1
    return level



