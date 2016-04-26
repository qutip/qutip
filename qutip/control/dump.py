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
         self.ctrl_amps = None
         self.dyn_gen = None
         self.prop = None
         self.prop_grad = None
         self.fwd_evo = None
         self.onwd_evo = None
         self.onto_evo = None
         
    def writeout(self, f=None):
        """ write all the objects out to files """
        fall = None
        if hasattr(f, 'write'):
            # write all to this stream
            fall = f
        elif f:
            fall = open(f, 'w')
        # otherwise files for each type will be created
        
        
         