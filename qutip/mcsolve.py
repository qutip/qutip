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
#Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson
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
from qutip.cyQ.cy_mc_funcs import mc_expect,spmv,spmv1d
from qutip.cyQ.ode_rhs import cyq_ode_rhs
from qutip.cyQ.codegen import Codegen
from Odedata import Odedata
from odechecks import _ode_checks
import qutip.settings
from _reset import _reset_odeconfig

def mcsolve(H,psi0,tlist,c_ops,e_ops,ntraj=500,args={},options=Odeoptions()):
    """Monte-Carlo evolution of a state vector :math:`|\psi \\rangle` for a given
    Hamiltonian and sets of collapse operators, and possibly, operators
    for calculating expectation values. Options for the underlying ODE solver are 
    given by the Odeoptions class.
    
    mcsolve supports time-dependent Hamiltonians and collapse operators using either
    Python functions of strings to represent time-dependent coefficients.  Note that, 
    the system Hamiltonian MUST have at least one constant term.
    
    As an example of a time-dependent problem, consider a Hamiltonian with two terms ``H0``
    and ``H1``, where ``H1`` is time-dependent with coefficient ``sin(w*t)``, and collapse operators
    ``C0`` and ``C1``, where ``C1`` is time-dependent with coeffcient ``exp(-a*t)``.  Here, w and a are
    constant arguments with values ``W`` and ``A``.  
    
    Using the Python function time-dependent format requires two Python functions,
    one for each collapse coefficient. Therefore, this problem could be expressed as::
    
        def H1_coeff(t,args):
            return sin(args['w']*t)
    
        def C1_coeff(t,args):
            return exp(-args['a']*t)
    
        H=[H0,[H1,H1_coeff]]
    
        c_op_list=[C0,[C1,C1_coeff]]
    
        args={'a':A,'w':W}
    
    or in String (Cython) format we could write::
    
        H=[H0,[H1,'sin(w*t)']]
    
        c_op_list=[C0,[C1,'exp(-a*t)']]
    
        args={'a':A,'w':W}
    
    Constant terms are preferably placed first in the Hamiltonian and collapse 
    operator lists.
    
    Parameters
    ----------
    H : qobj
        System Hamiltonian.
    psi0 : qobj 
        Initial state vector
    tlist : array_like 
        Times at which results are recorded.
    ntraj : int 
        Number of trajectories to run.
    c_ops : array_like 
        ``list`` or ``array`` of collapse operators.
    e_ops : array_like 
        ``list`` or ``array`` of operators for calculating expectation values.
    args : dict
        Arguments for time-dependent Hamiltonian and collapse operator terms.
    options : Odeoptions
        Instance of ODE solver options.
    
    Returns
    -------
    results : Odedata    
        Object storing all results from simulation.
        
    """
    if psi0.type!='ket':
        raise Exception("Initial state must be a state vector.")
    odeconfig.options=options
    #set num_cpus to the value given in qutip.settings if none in Odeoptions
    if not odeconfig.options.num_cpus:
        odeconfig.options.num_cpus=qutip.settings.num_cpus
    #set initial value data
    if options.tidy:
        odeconfig.psi0=psi0.tidyup(options.atol).full()
    else:
        odeconfig.psi0=psi0.full()
    odeconfig.psi0_dims=psi0.dims
    odeconfig.psi0_shape=psi0.shape
    #----
    
    #----------------------------------------------
    # SETUP ODE DATA IF NONE EXISTS OR NOT REUSING
    #----------------------------------------------
    if (not options.rhs_reuse) or (not odeconfig.tdfunc):
        #reset odeconfig collapse and time-dependence flags to default values
        _reset_odeconfig()
        
        #set general items
        odeconfig.tlist=tlist
        if isinstance(ntraj,(list,ndarray)):
            odeconfig.ntraj=sort(ntraj)[-1]
        else:
            odeconfig.ntraj=ntraj
        
        #check for type of time-dependence (if any)
        time_type,h_stuff,c_stuff=_ode_checks(H,c_ops,'mc')
        h_terms=len(h_stuff[0])+len(h_stuff[1])+len(h_stuff[2])
        c_terms=len(c_stuff[0])+len(c_stuff[1])+len(c_stuff[2])
        #set time_type for use in multiprocessing
        odeconfig.tflag=time_type
        
        #-Check for PyObjC on Mac platforms
        if sys.platform=='darwin':
            try:
                import Foundation
            except:
                odeconfig.options.gui=False

        #check if running in iPython and using Cython compiling (then no GUI to work around error)
        if odeconfig.options.gui and odeconfig.tflag in array([1,10,11]):
            try:
                __IPYTHON__
            except:
                pass
            else:
                odeconfig.options.gui=False    
        if qutip.settings.qutip_gui=="NONE":
            odeconfig.options.gui=False

        #check for collapse operators
        if c_terms>0:
            odeconfig.cflag=1
        else:
            odeconfig.cflag=0
    
        #Configure data
        _mc_data_config(H,psi0,h_stuff,c_ops,c_stuff,args,e_ops,options)
        
        if odeconfig.tflag in array([1,10,11]): #compile time-depdendent RHS code
            os.environ['CFLAGS'] = '-w'
            import pyximport
            pyximport.install(setup_args={'include_dirs':[numpy.get_include()]})
            if odeconfig.tflag in array([1,11]):
                code = compile('from '+odeconfig.tdname+' import cyq_td_ode_rhs,col_spmv,col_expect', '<string>', 'exec')
                exec(code)
                odeconfig.tdfunc=cyq_td_ode_rhs
                odeconfig.colspmv=col_spmv
                odeconfig.colexpect=col_expect
            else:
                code = compile('from '+odeconfig.tdname+' import cyq_td_ode_rhs', '<string>', 'exec')
                exec(code)
                odeconfig.tdfunc=cyq_td_ode_rhs
            try:
                os.remove(odeconfig.tdname+".pyx")
            except:
                print("Error removing pyx file.  File not found.")
        elif odeconfig.tflag==0:
            odeconfig.tdfunc=cyq_ode_rhs
    else:#setup args for new parameters when rhs_reuse=True and tdfunc is given
        #string based
        if odeconfig.tflag in array([1,10,11]):
            if any(args):
                odeconfig.c_args=[]
                arg_items=args.items()
                for k in xrange(len(args)):
                    odeconfig.c_args.append(arg_items[k][1])
        #function based
        elif odeconfig.tflag in array([2,3,20,22]):
            odeconfig.h_func_args=args
    
    
    #load monte-carlo class
    mc=MC_class()
    #RUN THE SIMULATION
    mc.run()
    
    
    #AFTER MCSOLVER IS DONE --------------------------------------
    
    
    
    #-------COLLECT AND RETURN OUTPUT DATA IN ODEDATA OBJECT --------------#
    output=Odedata()
    output.solver='mcsolve'
    #state vectors
    if any(mc.psi_out) and odeconfig.options.mc_avg:
        output.states=mc.psi_out
    #expectation values
    
    if any(mc.expect_out) and odeconfig.cflag and odeconfig.options.mc_avg:#averaging if multiple trajectories
        if isinstance(ntraj,int):
            output.expect=mean(mc.expect_out,axis=0)
        elif isinstance(ntraj,(list,ndarray)):
            output.expect=[]
            for num in ntraj:
                expt_data=mean(mc.expect_out[:num],axis=0)
                data_list=[]
                if any([op.isherm==False for op in e_ops]):
                    for k in xrange(len(e_ops)):
                        if e_ops[k].isherm:
                            data_list.append(real(expt_data[k]))
                        else:
                            data_list.append(expt_data[k])
                else:
                    data_list=[data for data in expt_data]
                output.expect.append(data_list)
    else:#no averaging for single trajectory or if mc_avg flag (Odeoptions) is off
        output.expect=mc.expect_out

    #simulation parameters
    output.times=odeconfig.tlist
    output.num_expect=odeconfig.e_num
    output.num_collapse=odeconfig.c_num
    output.ntraj=odeconfig.ntraj
    output.collapse_times=mc.collapse_times_out
    output.collapse_which=mc.which_op_out
    return output


#--------------------------------------------------------------
# MONTE-CARLO CLASS                                           #
#--------------------------------------------------------------
class MC_class():
    """
    Private class for solving Monte-Carlo evolution from mcsolve
    
    """
    def __init__(self):
        
        #-----------------------------------#
        # INIT MC CLASS
        #-----------------------------------#
    
        #----MAIN OBJECT PROPERTIES--------------------#
        ##holds instance of the ProgressBar class
        self.bar=None
        ##holds instance of the Pthread class
        self.thread=None
        #Number of completed trajectories
        self.count=0
        ##step-size for count attribute
        self.step=1
        ##Percent of trajectories completed
        self.percent=0.0
        ##used in implimenting the command line progress ouput
        self.level=0.1
        ##times at which to output state vectors or expectation values
        ##number of time steps in tlist
        self.num_times=len(odeconfig.tlist)
        #holds seed for random number generator
        self.seed=None
        #holds expected time to completion
        self.st=None
        #number of cpus to be used 
        self.cpus=odeconfig.options.num_cpus
        #set output variables, even if they are not used to simplify output code.
        self.psi_out=None
        self.expect_out=None
        self.collapse_times_out=None
        self.which_op_out=None
        
        #FOR EVOLUTION FOR NO COLLAPSE OPERATORS
        if odeconfig.c_num==0:
            if odeconfig.e_num==0:
                ##Output array of state vectors calculated at times in tlist
                self.psi_out=array([Qobj()]*self.num_times)#preallocate array of Qobjs
            elif odeconfig.e_num!=0:#no collpase expectation values
                ##List of output expectation values calculated at times in tlist
                self.expect_out=[]
                for i in xrange(odeconfig.e_num):
                    if odeconfig.e_ops_isherm[i]:#preallocate real array of zeros
                        self.expect_out.append(zeros(self.num_times))
                    else:#preallocate complex array of zeros
                        self.expect_out.append(zeros(self.num_times,dtype=complex))
                    self.expect_out[i][0]=mc_expect(odeconfig.e_ops_data[i],odeconfig.e_ops_ind[i],odeconfig.e_ops_ptr[i],odeconfig.e_ops_isherm[i],odeconfig.psi0)
        
        #FOR EVOLUTION WITH COLLAPSE OPERATORS
        elif odeconfig.c_num!=0:
            #preallocate #ntraj arrays for state vectors, collapse times, and which operator
            self.collapse_times_out=zeros((odeconfig.ntraj),dtype=ndarray)
            self.which_op_out=zeros((odeconfig.ntraj),dtype=ndarray)
            if odeconfig.e_num==0:# if no expectation operators, preallocate #ntraj arrays for state vectors
                self.psi_out=array([zeros((self.num_times),dtype=object) for q in xrange(odeconfig.ntraj)])#preallocate array of Qobjs
            else: #preallocate array of lists for expectation values
                self.expect_out=[[] for x in xrange(odeconfig.ntraj)]
    
    
    #-------------------------------------------------#
    # CLASS METHODS
    #-------------------------------------------------#
    def callback(self,results):
        r=results[0]
        if odeconfig.e_num==0:#output state-vector
            self.psi_out[r]=results[1]
        else:#output expectation values
            self.expect_out[r]=results[1]
        self.collapse_times_out[r]=results[2]
        self.which_op_out[r]=results[3]
        self.count+=self.step
        if (not odeconfig.options.gui): #do not use GUI
            self.percent=self.count/(1.0*odeconfig.ntraj)
            if self.count/float(odeconfig.ntraj)>=self.level:
                #calls function to determine simulation time remaining
                self.level=_time_remaining(self.st,odeconfig.ntraj,self.count,self.level)
    #-----
    def parallel(self,args,top=None):  
        self.st=datetime.datetime.now() #set simulation starting time
        pl=Pool(processes=self.cpus)
        [pl.apply_async(mc_alg_evolve,args=(nt,args),callback=top.callback) for nt in xrange(0,odeconfig.ntraj)]
        pl.close()
        try:
            pl.join()
        except KeyboardInterrupt:
            print "Cancel all MC threads on keyboard interrupt"
            pl.terminate()
            pl.join()
        return
    #-----
    def run(self):
        
        if odeconfig.c_num==0:
            if odeconfig.ntraj!=1:#check if ntraj!=1 which is pointless for no collapse operators
                odeconfig.ntraj=1
                print('No collapse operators specified.\nRunning a single trajectory only.\n')
            if odeconfig.e_num==0:# return psi Qobj at each requested time 
                self.psi_out=no_collapse_psi_out(odeconfig.options,odeconfig.psi0,odeconfig.tlist,self.num_times,odeconfig.psi0_dims,odeconfig.psi0_shape,self.psi_out)
            else:# return expectation values of requested operators
                self.expect_out=no_collapse_expect_out(odeconfig.options,odeconfig.psi0,odeconfig.tlist,odeconfig.e_ops_data,odeconfig.e_ops_ind,odeconfig.e_ops_ptr,odeconfig.e_ops_isherm,self.num_times,odeconfig.psi0_dims,odeconfig.psi0_shape,self.expect_out)
        elif odeconfig.c_num!=0:
            self.seed=array([int(ceil(random.rand()*1e4)) for ll in xrange(odeconfig.ntraj)])
            if odeconfig.e_num==0:
                mc_alg_out=zeros((self.num_times),dtype=ndarray)
                mc_alg_out[0]=odeconfig.psi0
            else:
                #PRE-GENERATE LIST OF EXPECTATION VALUES
                mc_alg_out=[]
                for i in xrange(odeconfig.e_num):
                    if odeconfig.e_ops_isherm[i]:#preallocate real array of zeros
                        mc_alg_out.append(zeros(self.num_times))
                    else:#preallocate complex array of zeros
                        mc_alg_out.append(zeros(self.num_times,dtype=complex))
                    mc_alg_out[i][0]=mc_expect(odeconfig.e_ops_data[i],odeconfig.e_ops_ind[i],odeconfig.e_ops_ptr[i],odeconfig.e_ops_isherm[i],odeconfig.psi0)
            
            #set arguments for input to monte-carlo
            args=(mc_alg_out,odeconfig.options,odeconfig.tlist,self.num_times,self.seed)
            if not odeconfig.options.gui:
                self.parallel(args,self)
            else:
                if qutip.settings.qutip_gui=="PYSIDE":
                    from PySide import QtGui,QtCore
                elif qutip.settings.qutip_gui=="PYQT4":
                    from PyQt4 import QtGui,QtCore
                from gui.ProgressBar import ProgressBar,Pthread
                app=QtGui.QApplication.instance()#checks if QApplication already exists (needed for iPython)
                if not app:#create QApplication if it doesnt exist
                    app = QtGui.QApplication(sys.argv)
                thread=Pthread(target=self.parallel,args=args,top=self)
                self.bar=ProgressBar(self,thread,odeconfig.ntraj,self.cpus)
                QtCore.QTimer.singleShot(0,self.bar.run)
                self.bar.show()
                self.bar.activateWindow()
                self.bar.raise_()
                app.exec_()
                return
                





#----------------------------------------------------
# CODES FOR PYTHON FUNCTION BASED TIME-DEPENDENT RHS
#----------------------------------------------------
#RHS of ODE for time-dependent systems with no collapse operators
def tdRHS(t,psi):
    h_data=odeconfig.h_func(t,odeconfig.h_func_args).data
    return spmv1d(-1.0j*h_data.data,h_data.indices,h_data.indptr,psi)

#RHS of ODE for constant Hamiltonian and at least one function based collapse operator
def cRHStd(t,psi):
    sys=cyq_ode_rhs(t,psi,odeconfig.h_data,odeconfig.h_ind,odeconfig.h_ptr)
    col=array([abs(odeconfig.c_funcs[j](t,odeconfig.c_func_args))**2*spmv1d(odeconfig.n_ops_data[j],odeconfig.n_ops_ind[j],odeconfig.n_ops_ptr[j],psi) for j in odeconfig.c_td_inds])
    return sys-0.5*sum(col,0)

#RHS of ODE for function-list based Hamiltonian
def tdRHStd(t,psi):
    const_term=spmv1d(odeconfig.h_data,odeconfig.h_ind,odeconfig.h_ptr,psi)
    h_func_term=array([odeconfig.h_funcs[j](t,odeconfig.h_func_args)*spmv1d(odeconfig.h_td_data[j],odeconfig.h_td_ind[j],odeconfig.h_td_ptr[j],psi) for j in odeconfig.h_td_inds])
    col_func_terms=array([abs(odeconfig.c_funcs[j](t,odeconfig.c_func_args))**2*spmv1d(odeconfig.n_ops_data[j],odeconfig.n_ops_ind[j],odeconfig.n_ops_ptr[j],psi) for j in odeconfig.c_td_inds])
    return const_term-1.0j*sum(h_func_term,0)-0.5*sum(col_func_terms,0)

#RHS of ODE for python function Hamiltonian
def pyRHSc(t,psi):
    h_func_data=odeconfig.h_funcs(t,odeconfig.h_func_args).data
    h_func_term=-1.0j*spmv1d(h_func_data.data,h_func_data.indices,h_func_data.indptr,psi)
    const_col_term=0
    if len(odeconfig.c_const_inds)>0:    
        const_col_term=spmv1d(odeconfig.h_data,odeconfig.h_ind,odeconfig.h_ptr,psi)
    return h_func_term+const_col_term
#----------------------------------------------------
# END PYTHON FUNCTION RHS
#----------------------------------------------------








######---return psi at requested times for no collapse operators---######
def no_collapse_psi_out(opt,psi_in,tlist,num_times,psi_dims,psi_shape,psi_out):
    ##Calculates state vectors at times tlist if no collapse AND no expectation values are given.
    #
    if odeconfig.tflag in array([1,10,11]):
        ODE=ode(odeconfig.tdfunc)
        code = compile('ODE.set_f_params('+odeconfig.string+')', '<string>', 'exec')
        exec(code)
    elif odeconfig.tflag==2:
        ODE=ode(cRHStd)
    elif odeconfig.tflag in array([20,22]):
        ODE=ode(tdRHStd)
    elif odeconfig.tflag==3:
        ODE=ode(pyRHSc)
    else:
        ODE = ode(cyq_ode_rhs)
        ODE.set_f_params(odeconfig.h_data, odeconfig.h_ind, odeconfig.h_ptr)
        
    ODE.set_integrator('zvode',method=opt.method,order=opt.order,atol=opt.atol,rtol=opt.rtol,nsteps=opt.nsteps,first_step=opt.first_step,min_step=opt.min_step,max_step=opt.max_step) #initialize ODE solver for RHS
    ODE.set_initial_value(psi_in,tlist[0]) #set initial conditions
    psi_out[0]=Qobj(psi_in,odeconfig.psi0_dims,odeconfig.psi0_shape,'ket')
    for k in xrange(1,num_times):
        ODE.integrate(tlist[k],step=0) #integrate up to tlist[k]
        if ODE.successful():
            psi_out[k]=Qobj(ODE.y/norm(ODE.y,2),odeconfig.psi0_dims,odeconfig.psi0_shape,'ket')
        else:
            raise ValueError('Error in ODE solver')
    return psi_out
#------------------------------------------------------------------------


######---return expectation values at requested times for no collapse operators---######
def no_collapse_expect_out(opt,psi_in,tlist,e_ops_data,e_ops_ind,e_ops_ptr,e_ops_isherm,num_times,psi_dims,psi_shape,expect_out):
    ##Calculates xpect.values at times tlist if no collapse ops. given
    #  
    #------------------------------------
    num_expect=len(e_ops_data)
    if odeconfig.tflag in array([1,10,11]):
        ODE=ode(odeconfig.tdfunc)
        code = compile('ODE.set_f_params('+odeconfig.string+')', '<string>', 'exec')
        exec(code)
    elif odeconfig.tflag==2:
        ODE=ode(cRHStd)
    elif odeconfig.tflag in array([20,22]):
        ODE=ode(tdRHStd)
    elif odeconfig.tflag==3:
        ODE=ode(pyRHSc)
    else:
        ODE = ode(cyq_ode_rhs)
        ODE.set_f_params(odeconfig.h_data, odeconfig.h_ind, odeconfig.h_ptr)
    ODE.set_integrator('zvode',method=opt.method,order=opt.order,atol=opt.atol,rtol=opt.rtol,nsteps=opt.nsteps,first_step=opt.first_step,min_step=opt.min_step,max_step=opt.max_step) #initialize ODE solver for RHS
    ODE.set_initial_value(psi_in,tlist[0]) #set initial conditions
    for jj in xrange(num_expect):
        expect_out[jj][0]=mc_expect(e_ops_data[jj],e_ops_ind[jj],e_ops_ptr[jj],e_ops_isherm[jj],psi_in)
    for k in xrange(1,num_times):
        ODE.integrate(tlist[k],step=0) #integrate up to tlist[k]
        if ODE.successful():
            state=ODE.y/norm(ODE.y)
            for jj in xrange(num_expect):
                expect_out[jj][k]=mc_expect(e_ops_data[jj],e_ops_ind[jj],e_ops_ptr[jj],e_ops_isherm[jj],state)
        else:
            raise ValueError('Error in ODE solver')
    return expect_out #return times and expectiation values
#------------------------------------------------------------------------


#---single-trajectory for monte-carlo---          
def mc_alg_evolve(nt,args):
    """
    Monte-Carlo algorithm returning state-vector or expectation values at times tlist for a single trajectory.
    """
    #get input data
    mc_alg_out,opt,tlist,num_times,seeds=args

    #number of operators of each type
    num_expect=odeconfig.e_num
    num_collapse=odeconfig.c_num
    
    collapse_times=[] #times at which collapse occurs
    which_oper=[] # which operator did the collapse
    
    #SEED AND RNG AND GENERATE
    random.seed(seeds[nt])
    rand_vals=random.rand(2)#first rand is collapse norm, second is which operator
    
    #CREATE ODE OBJECT CORRESPONDING TO DESIRED TIME-DEPENDENCE
    if odeconfig.tflag in array([1,10,11]):
        ODE=ode(odeconfig.tdfunc)
        code = compile('ODE.set_f_params('+odeconfig.string+')', '<string>', 'exec')
        exec(code)
    elif odeconfig.tflag==2:
        ODE=ode(cRHStd)
    elif odeconfig.tflag in array([20,22]):
        ODE=ode(tdRHStd)
    elif odeconfig.tflag==3:
        ODE=ode(pyRHSc)
    else:
        ODE = ode(cyq_ode_rhs)
        ODE.set_f_params(odeconfig.h_data, odeconfig.h_ind, odeconfig.h_ptr)

    #initialize ODE solver for RHS
    ODE.set_integrator('zvode',method=opt.method,order=opt.order,atol=opt.atol,rtol=opt.rtol,nsteps=opt.nsteps,
                        first_step=opt.first_step,min_step=opt.min_step,max_step=opt.max_step)
    
    #set initial conditions
    ODE.set_initial_value(odeconfig.psi0,tlist[0])
    
    #RUN ODE UNTIL EACH TIME IN TLIST
    cinds=arange(num_collapse)
    for k in xrange(1,num_times):
        #ODE WHILE LOOP FOR INTEGRATE UP TO TIME TLIST[k]
        while ODE.successful() and ODE.t<tlist[k]:
            last_t=ODE.t;last_y=ODE.y
            ODE.integrate(tlist[k],step=1) #integrate up to tlist[k], one step at a time.
            psi_nrm2=norm(ODE.y,2)**2
            if psi_nrm2<=rand_vals[0]:# <== collpase has occured
                collapse_times.append(ODE.t)
                #some string based collapse operators
                if odeconfig.tflag in array([1,11]):
                    n_dp=[mc_expect(odeconfig.n_ops_data[i],odeconfig.n_ops_ind[i],odeconfig.n_ops_ptr[i],1,ODE.y) for i in odeconfig.c_const_inds]
                    exec(odeconfig.col_expect_code) #calculates the expectation values for time-dependent norm collapse operators
                    n_dp=array(n_dp)
                
                #some Python function based collapse operators
                elif odeconfig.tflag in array([2,20,22]):
                    n_dp=[mc_expect(odeconfig.n_ops_data[i],odeconfig.n_ops_ind[i],odeconfig.n_ops_ptr[i],1,ODE.y) for i in odeconfig.c_const_inds]
                    n_dp+=[abs(odeconfig.c_funcs[i](ODE.t,odeconfig.c_func_args))**2*mc_expect(odeconfig.n_ops_data[i],odeconfig.n_ops_ind[i],odeconfig.n_ops_ptr[i],1,ODE.y) for i in odeconfig.c_td_inds]
                    n_dp=array(n_dp)
                #all constant collapse operators.
                else:    
                    n_dp=array([mc_expect(odeconfig.n_ops_data[i],odeconfig.n_ops_ind[i],odeconfig.n_ops_ptr[i],1,ODE.y) for i in xrange(num_collapse)])
                
                #determine which operator does collapse
                kk=cumsum(n_dp/sum(n_dp))
                j=cinds[kk>=rand_vals[1]][0]
                which_oper.append(j) #record which operator did collapse
                if j in odeconfig.c_const_inds:
                    state=spmv(odeconfig.c_ops_data[j],odeconfig.c_ops_ind[j],odeconfig.c_ops_ptr[j],ODE.y)
                else:
                    if odeconfig.tflag in array([1,11]):
                        exec(odeconfig.col_spmv_code)#calculates the state vector for  collapse by a time-dependent collapse operator
                    else:
                        state=odeconfig.c_funcs[j](ODE.t,odeconfig.c_func_args)*spmv(odeconfig.c_ops_data[j],odeconfig.c_ops_ind[j],odeconfig.c_ops_ptr[j],ODE.y)
                state_nrm=norm(state,2)
                ODE.set_initial_value(state/state_nrm,ODE.t)
                rand_vals=random.rand(2)
        #-------------------------------------------------------
        ###--after while loop--####
        psi=copy(ODE.y)
        if ODE.t>last_t:
            psi=(psi-last_y)/(ODE.t-last_t)*(tlist[k]-last_t)+last_y
        epsi=psi/norm(psi,2)
        if num_expect==0:
            mc_alg_out[k]=epsi
        else:
            for jj in xrange(num_expect):
                mc_alg_out[jj][k]=mc_expect(odeconfig.e_ops_data[jj],odeconfig.e_ops_ind[jj],odeconfig.e_ops_ptr[jj],odeconfig.e_ops_isherm[jj],epsi)
    #RETURN VALUES
    if num_expect==0:
        mc_alg_out=array([Qobj(k,odeconfig.psi0_dims,odeconfig.psi0_shape,fast='mc') for k in mc_alg_out])
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


def _mc_data_config(H,psi0,h_stuff,c_ops,c_stuff,args,e_ops,options):
    """Creates the appropriate data structures for the monte carlo solver
    based on the given time-dependent, or indepdendent, format.
    """
    
    #take care of expectation values, if any
    if any(e_ops):
        odeconfig.e_num=len(e_ops)
        for op in e_ops:
            if isinstance(op,list):
                op=op[0]
            odeconfig.e_ops_data.append(op.data.data)
            odeconfig.e_ops_ind.append(op.data.indices)
            odeconfig.e_ops_ptr.append(op.data.indptr)
            odeconfig.e_ops_isherm.append(op.isherm)
        
        odeconfig.e_ops_data=array(odeconfig.e_ops_data)
        odeconfig.e_ops_ind=array(odeconfig.e_ops_ind)
        odeconfig.e_ops_ptr=array(odeconfig.e_ops_ptr)
        odeconfig.e_ops_isherm=array(odeconfig.e_ops_isherm)
    #----
    
    #take care of collapse operators, if any
    if any(c_ops):
        odeconfig.c_num=len(c_ops)
        for c_op in c_ops:
            if isinstance(c_op,list):
                c_op=c_op[0]
            n_op=c_op.dag()*c_op
            odeconfig.c_ops_data.append(c_op.data.data)
            odeconfig.c_ops_ind.append(c_op.data.indices)
            odeconfig.c_ops_ptr.append(c_op.data.indptr)
            #norm ops
            odeconfig.n_ops_data.append(n_op.data.data)
            odeconfig.n_ops_ind.append(n_op.data.indices)
            odeconfig.n_ops_ptr.append(n_op.data.indptr)
        #to array
        odeconfig.c_ops_data=array(odeconfig.c_ops_data)
        odeconfig.c_ops_ind=array(odeconfig.c_ops_ind)
        odeconfig.c_ops_ptr=array(odeconfig.c_ops_ptr)
        
        odeconfig.n_ops_data=array(odeconfig.n_ops_data)
        odeconfig.n_ops_ind=array(odeconfig.n_ops_ind)
        odeconfig.n_ops_ptr=array(odeconfig.n_ops_ptr)
    #----
    
    
    #--------------------------------------------
    # START CONSTANT H & C_OPS CODE
    #--------------------------------------------
    if odeconfig.tflag==0:
        if odeconfig.cflag:
            odeconfig.c_const_inds=arange(len(c_ops))
            for c_op in c_ops:
                n_op=c_op.dag()*c_op
                H -= 0.5j * n_op #combine Hamiltonian and collapse terms into one
        #construct Hamiltonian data structures
        if options.tidy:
            H=H.tidyup(options.atol)
        odeconfig.h_data=-1.0j*H.data.data
        odeconfig.h_ind=H.data.indices
        odeconfig.h_ptr=H.data.indptr  
    #----
    
    #--------------------------------------------
    # START STRING BASED TIME-DEPENDENCE
    #--------------------------------------------
    elif odeconfig.tflag in array([1,10,11]):
        #take care of arguments for collapse operators, if any
        if any(args):
            arg_items=args.items()
            for k in xrange(len(args)):
                odeconfig.c_args.append(arg_items[k][1])
        #constant Hamiltonian / string-type collapse operators
        if odeconfig.tflag==1:
            H_inds=arange(1)
            H_tdterms=0
            len_h=1
            C_inds=arange(odeconfig.c_num)
            C_td_inds=array(c_stuff[2]) #find inds of time-dependent terms
            C_const_inds=setdiff1d(C_inds,C_td_inds) #find inds of constant terms
            C_tdterms=[c_ops[k][1] for k in C_td_inds] #extract time-dependent coefficients (strings)
            odeconfig.c_const_inds=C_const_inds#store indicies of constant collapse terms
            odeconfig.c_td_inds=C_td_inds#store indicies of time-dependent collapse terms
            
            for k in odeconfig.c_const_inds:
                H-=0.5j*(c_ops[k].dag()*c_ops[k])
            if options.tidy:
                H=H.tidyup(options.atol)
            odeconfig.h_data=[H.data.data]
            odeconfig.h_ind=[H.data.indices]
            odeconfig.h_ptr=[H.data.indptr]
            for k in odeconfig.c_td_inds:
                op=c_ops[k][0].dag()*c_ops[k][0]
                odeconfig.h_data.append(-0.5j*op.data.data)
                odeconfig.h_ind.append(op.data.indices)
                odeconfig.h_ptr.append(op.data.indptr)
            odeconfig.h_data=-1.0j*array(odeconfig.h_data)
            odeconfig.h_ind=array(odeconfig.h_ind)
            odeconfig.h_ptr=array(odeconfig.h_ptr)
            #--------------------------------------------
            # END OF IF STATEMENT
            #--------------------------------------------
        
        
        #string-type Hamiltonian & at least one string-type collapse operator
        else:
            H_inds=arange(len(H))
            H_td_inds=array(h_stuff[2]) #find inds of time-dependent terms
            H_const_inds=setdiff1d(H_inds,H_td_inds) #find inds of constant terms
            H_tdterms=[H[k][1] for k in H_td_inds] #extract time-dependent coefficients (strings or functions)
            H=array([sum(H[k] for k in H_const_inds)]+[H[k][0] for k in H_td_inds]) #combine time-INDEPENDENT terms into one.
            len_h=len(H)
            H_inds=arange(len_h)
            odeconfig.h_td_inds=arange(1,len_h)#store indicies of time-dependent Hamiltonian terms
            #if there are any collpase operators
            if odeconfig.c_num>0:
                if odeconfig.tflag==10: #constant collapse operators
                    odeconfig.c_const_inds=arange(odeconfig.c_num)
                    for k in odeconfig.c_const_inds:
                        H[0]-=0.5j*(c_ops[k].dag()*c_ops[k])
                    C_inds=arange(odeconfig.c_num)
                    C_tdterms=array([])
                #-----
                else:#some time-dependent collapse terms
                    C_inds=arange(odeconfig.c_num)
                    C_td_inds=array(c_stuff[2]) #find inds of time-dependent terms
                    C_const_inds=setdiff1d(C_inds,C_td_inds) #find inds of constant terms
                    C_tdterms=[c_ops[k][1] for k in C_td_inds] #extract time-dependent coefficients (strings)
                    odeconfig.c_const_inds=C_const_inds#store indicies of constant collapse terms
                    odeconfig.c_td_inds=C_td_inds#store indicies of time-dependent collapse terms
                    for k in odeconfig.c_const_inds:
                        H[0]-=0.5j*(c_ops[k].dag()*c_ops[k])
            else:#set empty objects if no collapse operators
                C_const_inds=arange(odeconfig.c_num)
                odeconfig.c_const_inds=arange(odeconfig.c_num)
                odeconfig.c_td_inds=array([])
                C_tdterms=array([])
                C_inds=array([])
            
            #tidyup
            if options.tidy:
                H=array([H[k].tidyup(options.atol) for k in xrange(len_h)])
            #construct data sets
            odeconfig.h_data=[H[k].data.data for k in xrange(len_h)]
            odeconfig.h_ind=[H[k].data.indices for k in xrange(len_h)]
            odeconfig.h_ptr=[H[k].data.indptr for k in xrange(len_h)]
            for k in odeconfig.c_td_inds:
                odeconfig.h_data.append(-0.5j*odeconfig.n_ops_data[k])
                odeconfig.h_ind.append(odeconfig.n_ops_ind[k])
                odeconfig.h_ptr.append(odeconfig.n_ops_ptr[k])
            odeconfig.h_data=-1.0j*array(odeconfig.h_data)
            odeconfig.h_ind=array(odeconfig.h_ind)
            odeconfig.h_ptr=array(odeconfig.h_ptr)
            #--------------------------------------------
            # END OF ELSE STATEMENT
            #--------------------------------------------
        
        #set execuatble code for collapse expectation values and spmv
        col_spmv_code="state=odeconfig.colspmv(j,ODE.t,odeconfig.c_ops_data[j],odeconfig.c_ops_ind[j],odeconfig.c_ops_ptr[j],ODE.y"
        col_expect_code="n_dp+=[odeconfig.colexpect(i,ODE.t,odeconfig.n_ops_data[i],odeconfig.n_ops_ind[i],odeconfig.n_ops_ptr[i],ODE.y"
        for kk in range(len(odeconfig.c_args)):
            col_spmv_code+=",odeconfig.c_args["+str(kk)+"]"
            col_expect_code+=",odeconfig.c_args["+str(kk)+"]"
        col_spmv_code+=")"
        col_expect_code+=") for i in odeconfig.c_td_inds]"
        odeconfig.col_spmv_code=compile(col_spmv_code,'<string>', 'exec')
        odeconfig.col_expect_code=compile(col_expect_code,'<string>', 'exec')    
        #----
        
        #setup ode args string
        odeconfig.string=""
        data_range=range(len(odeconfig.h_data))
        for k in data_range:
            odeconfig.string+="odeconfig.h_data["+str(k)+"],odeconfig.h_ind["+str(k)+"],odeconfig.h_ptr["+str(k)+"]"
            if k!=data_range[-1]:
                odeconfig.string+="," 
        #attach args to ode args string
        if any(odeconfig.c_args):
            for kk in range(len(odeconfig.c_args)):
                odeconfig.string+=","+"odeconfig.c_args["+str(kk)+"]"
        #----
        name="rhs"+str(odeconfig.cgen_num)
        odeconfig.tdname=name
        cgen=Codegen(H_inds,H_tdterms,odeconfig.h_td_inds,args,C_inds,C_tdterms,odeconfig.c_td_inds,type='mc')
        cgen.generate(name+".pyx")
        #----
    #--------------------------------------------
    # END OF STRING TYPE TIME DEPENDENT CODE
    #--------------------------------------------
    
    #--------------------------------------------
    # START PYTHON FUNCTION BASED TIME-DEPENDENCE
    #--------------------------------------------
    elif odeconfig.tflag in array([2,20,22]):
        
        #take care of Hamiltonian
        if odeconfig.tflag==2:# constant Hamiltonian, at least one function based collapse operators
            H_inds=array([0])
            H_tdterms=0
            len_h=1
        else:# function based Hamiltonian
            H_inds=arange(len(H))
            H_td_inds=array(h_stuff[1]) #find inds of time-dependent terms
            H_const_inds=setdiff1d(H_inds,H_td_inds) #find inds of constant terms    
            odeconfig.h_funcs=array([H[k][1] for k in H_td_inds])
            odeconfig.h_func_args=args
            Htd=array([H[k][0] for k in H_td_inds])
            odeconfig.h_td_inds=arange(len(Htd))
            H=sum(H[k] for k in H_const_inds)
        
        #take care of collapse operators
        C_inds=arange(odeconfig.c_num)
        C_td_inds=array(c_stuff[1]) #find inds of time-dependent terms
        C_const_inds=setdiff1d(C_inds,C_td_inds) #find inds of constant terms
        odeconfig.c_const_inds=C_const_inds#store indicies of constant collapse terms
        odeconfig.c_td_inds=C_td_inds#store indicies of time-dependent collapse terms    
        odeconfig.c_funcs=zeros(odeconfig.c_num,dtype=FunctionType)
        for k in odeconfig.c_td_inds:
            odeconfig.c_funcs[k]=c_ops[k][1]
        odeconfig.c_func_args=args
            
        #combine constant collapse terms with constant H and construct data
        for k in odeconfig.c_const_inds:
            H-=0.5j*(c_ops[k].dag()*c_ops[k])
        if options.tidy:
            H=H.tidyup(options.atol)
            Htd=array([Htd[j].tidyup(options.atol) for j in odeconfig.h_td_inds])
            #setup cosntant H terms data
        odeconfig.h_data=-1.0j*H.data.data
        odeconfig.h_ind=H.data.indices
        odeconfig.h_ptr=H.data.indptr     
        
        #setup td H terms data
        odeconfig.h_td_data=array([-1.0j*Htd[k].data.data for k in odeconfig.h_td_inds])
        odeconfig.h_td_ind=array([Htd[k].data.indices for k in odeconfig.h_td_inds])
        odeconfig.h_td_ptr=array([Htd[k].data.indptr for k in odeconfig.h_td_inds])
        #--------------------------------------------
        # END PYTHON FUNCTION BASED TIME-DEPENDENCE
        #--------------------------------------------
     
     
    #--------------------------------------------
    # START PYTHON FUNCTION BASED HAMILTONIAN
    #--------------------------------------------
    elif odeconfig.tflag==3:
         #take care of Hamiltonian
         odeconfig.h_funcs=H
         odeconfig.h_func_args=args
         
         #take care of collapse operators
         odeconfig.c_const_inds=arange(odeconfig.c_num)
         odeconfig.c_td_inds=array([]) #find inds of time-dependent terms 
         if len(odeconfig.c_const_inds)>0:
             H=0
             for k in odeconfig.c_const_inds:
                 H-=0.5j*(c_ops[k].dag()*c_ops[k])
             if options.tidy:
                 H=H.tidyup(options.atol)
             odeconfig.h_data=-1.0j*H.data.data
             odeconfig.h_ind=H.data.indices
             odeconfig.h_ptr=H.data.indptr
        
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         