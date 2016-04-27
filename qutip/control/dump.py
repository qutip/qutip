# -*- coding: utf-8 -*-
# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2016 and later, Alexander J G Pitchford
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

"""
Classes that enable the storing of historical objects created during the 
pulse optimisation.
These are intented for debugging.
See the optimizer and dynamics objects for instrutcions on how to enable
data dumping.
"""

import os
import numpy as np
import copy
# QuTiP logging
import qutip.logging_utils
logger = qutip.logging_utils.get_logger()
import qutip.control.io as qtrlio
# QuTiP control modules
import qutip.control.dynamics as dynamics

DUMP_DIR = "qtrl_dump"

class Dump(object):
    """
    A container for dump items.
    This lists for dump items is depends on the type
    """
    def init(self):
        self.reset()
        
    def reset(self):
        self.dump_dir = None
        self.dump_file_ext = "txt"
        self.dump_to_file = False
        
    def create_dump_dir(self):
        """
        Checks test_out directory exists, creates it if not
        """
        if self.dump_dir is None or len(self.dump_dir) == 0:
            self.dump_dir = DUMP_DIR

        dir_ok, self.dump_dir, msg = qtrlio.create_dir(
                    self.dump_dir, desc='dump')

        if not dir_ok:
            self.dump_to_file = False
            msg += "\ndump file output will be suppressed."
            logger.error(msg)

        return dir_ok
        
class DynamicsDump(Dump):
    """
    A container for dumps of dynamics data.
    Mainly time evolution calculations
    """
    def init(self, dyn):
        if not isinstance(dyn, dynamics.Dynamics):
            raise TypeError("Must instantiate with {} type".format(
                                        dynamics.Dynamics))
        self.parent = dyn
        self.reset()
        
    def reset(self):
        dyn = self.parent
        self.evo_dumps = []
        self.dump_amps = True
        self.dump_dyn_gen = True
        self.dump_prop = True
        self.dump_prop_grad = True
        self.dump_fwd_evo = True
        self.dump_onwd_evo = dyn.fid_computer.uses_onwd_evo
        self.dump_onto_evo = dyn.fid_computer.uses_onto_evo
        
        self.fname_base = 'dyndump'
        
        
        
    def clear(self):
        self.evo_dumps.clear()
        
    def add_evo_dump(self):
        """
        Add dump of current time evolution generating objects
        """
        dyn = self.parent
        item = EvoCompDumpItem()
        item.idx = len(self.evo_dumps)
        self.evo_dumps.append()
        if self.dump_amps:
            item.ctrl_amps = copy.deepcopy(dyn.ctrl_amps)
        if self.dump_dyn_gen:
            item.dyn_gen = copy.deepcopy(dyn._dyn_gen)
        if self.dump_prop:
            item.prop = copy.deepcopy(dyn._prop)
        if self.dump_prop_grad:
            item.prop_grad = copy.deepcopy(dyn._prop_grad)
        if self.dump_fwd_evo:
            item.fwd_evo = copy.deepcopy(dyn._fwd_evo)
        if self.dump_onwd_evo:
            item.onwd_evo = copy.deepcopy(dyn._onwd_evo)
        if self.dump_onwd_evo:
            item.onto_evo = copy.deepcopy(dyn._onto_evo)    
            
class DumpItem(object):
    """
    An item in a dump list
    """
    def init(self):
        pass

class EvoCompDumpItem(DumpItem):
    """
    A copy of all objects generated to calculation one time evolution
    """
    def init(self, dump):
        if not isinstance(dump, DynamicsDump):
            raise TypeError("Must instantiate with {} type".format(
                                        DynamicsDump))
        self.parent = dump
        self.reset()
    
    def reset(self):
        self.idx = None
#        self.num_ctrls = None
#        self.num_tslots = None
        self.ctrl_amps = None
        self.dyn_gen = None
        self.prop = None
        self.prop_grad = None
        self.fwd_evo = None
        self.onwd_evo = None
        self.onto_evo = None
                
    def writeout(self, f=None):
        """ write all the objects out to files """
        dump = self.parent
        fall = None
        closefall = True
        closef = False
        # If specific file given then write everything to it
        if hasattr(f, 'write'):
            # write all to this stream
            fall = f
            closefall = False
            f.write("EVOLUTION COMPUTATION {}".format(self.idx))
        elif f:
            fall = open(f, 'w')
        else:   
            # otherwise files for each type will be created
            fname_base = "{}-evo{}".format(dump.fname_base, self.idx)
        
        #ctrl amps
        if self.ctrl_amps:
            if fall:
                f = fall
                f.write("Ctrl amps")
            else:
                fname = "{}-ctrl_amps.{}".format(fname_base, 
                                                dump.dump_file_ext)
                f = open(os.path.join(dump.dump_dir, fname), 'w')
                closef = True
            np.savetxt(f, self.ctrl_amps)
            if closef: f.close()
                
        # dynamics generators
        if self.dyn_gen:
            k = 0
            if fall:
                f = fall
                f.write("Dynamics Generators")
            else:
                fname = "{}-dyn_gen.{}".format(fname_base, 
                                                dump.dump_file_ext)
                f = open(os.path.join(dump.dump_dir, fname), 'w')
                closef = True
            for dg in self.dyn_gen:
                f.write("dynamics generator for timeslot {}".format(k))
                np.savetxt(f, self.dyn_gen[k])
                k += 1
            if closef: f.close()

        # Propagators
        if self.prop:
            k = 0
            if fall:
                f = fall
                f.write("Propagators")
            else:
                fname = "{}-prop.{}".format(fname_base, 
                                                dump.dump_file_ext)
                f = open(os.path.join(dump.dump_dir, fname), 'w')
                closef = True
            for dg in self.dyn_gen:
                f.write("Propagator for timeslot {}".format(k))
                np.savetxt(f, self.prop[k])
                k += 1
            if closef: f.close()
                
        # Propagator gradient
        if self.prop_grad:
            k = 0
            if fall:
                f = fall
                f.write("Propagator gradients")
            else:
                fname = "{}-prop_grad.{}".format(fname_base, 
                                                dump.dump_file_ext)
                f = open(os.path.join(dump.dump_dir, fname), 'w')
                closef = True
            for k in range(self.prop_grad.shape[0]):
                for j in range(self.prop_grad.shape[1]):
                    f.write("Propagator gradient for timeslot {} "
                            "control {}".format(k, j))
                    np.savetxt(f, self.prop_grad[k, j])
            if closef: f.close()

        # forward evolution
        if self.fwd_evo:
            k = 0
            if fall:
                f = fall
                f.write("Forward evolution")
            else:
                fname = "{}-fwd_evo.{}".format(fname_base, 
                                                dump.dump_file_ext)
                f = open(os.path.join(dump.dump_dir, fname), 'w')
                closef = True
            for dg in self.dyn_gen:
                f.write("Evolution from 0 to {}".format(k))
                np.savetxt(f, self.fwd_evo[k])
                k += 1
            if closef: f.close()
            
        # onward evolution
        if self.onwd_evo:
            k = 0
            if fall:
                f = fall
                f.write("Onward evolution")
            else:
                fname = "{}-onwd_evo.{}".format(fname_base, 
                                                dump.dump_file_ext)
                f = open(os.path.join(dump.dump_dir, fname), 'w')
                closef = True
            for dg in self.dyn_gen:
                f.write("Evolution from {} to end".format(k))
                np.savetxt(f, self.fwd_evo[k])
                k += 1
            if closef: f.close()
                
        # onto evolution
        if self.onto_evo:
            k = 0
            if fall:
                f = fall
                f.write("Onto evolution")
            else:
                fname = "{}-onto_evo.{}".format(fname_base, 
                                                dump.dump_file_ext)
                f = open(os.path.join(dump.dump_dir, fname), 'w')
                closef = True
            for dg in self.dyn_gen:
                f.write("Evolution from {} onto target".format(k))
                np.savetxt(f, self.fwd_evo[k])
                k += 1
            if closef: f.close()
            
                
        if closefall:
            fall.close()
        
        
         