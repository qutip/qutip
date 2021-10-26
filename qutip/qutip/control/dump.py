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
# QuTiP control modules
import qutip.control.io as qtrlio
from numpy.compat import asbytes

DUMP_DIR = "~/.qtrl_dump"

def _is_string(var):
    try:
        if isinstance(var, basestring):
            return True
    except NameError:
        try:
            if isinstance(var, str):
                return True
        except:
            return False
    except:
        return False

    return False

class Dump(object):
    """
    A container for dump items.
    The lists for dump items is depends on the type
    Note: abstract class

    Attributes
    ----------
    parent : some control object (Dynamics or Optimizer)
        aka the host. Object that generates the data that is dumped and is
        host to this dump object.

    dump_dir : str
        directory where files (if any) will be written out
        the path and be relative or absolute
        use ~/ to specify user home directory
        Note: files are only written when write_to_file is True
        of writeout is called explicitly
        Defaults to ~/.qtrl_dump

    level : string
        level of data dumping: SUMMARY, FULL or CUSTOM
        See property docstring for details
        Set automatically if dump is created by the setting host dumping attrib

    write_to_file : bool
        When set True data and summaries (as configured) will be written
        interactively to file during the processing
        Set during instantiation by the host based on its dump_to_file attrib

    dump_file_ext : str
        Default file extension for any file names that are auto generated

    fname_base : str
        First part of any auto generated file names.
        This is usually overridden in the subclass

    dump_summary : bool
        If True a summary is recorded each time a new item is added to the
        the dump.
        Default is True

    summary_sep : str
        delimiter for the summary file.
        default is a space

    data_sep : str
        delimiter for the data files (arrays saved to file).
        default is a space

    summary_file : str
        File path for summary file.
        Automatically generated. Can be set specifically

    """
    def __init__(self):
        self.reset()

    def reset(self):
        if self.parent:
            self.log_level = self.parent.log_level
            self.write_to_file = self.parent.dump_to_file
        else:
            self.write_to_file = False
        self._dump_dir = None
        self.dump_file_ext = "txt"
        self._fname_base = 'dump'
        self.dump_summary = True
        self.summary_sep = ' '
        self.data_sep = ' '
        self._summary_file_path = None
        self._summary_file_specified = False

    @property
    def log_level(self):
        return logger.level

    @log_level.setter
    def log_level(self, lvl):
        """
        Set the log_level attribute and set the level of the logger
        that is call logger.setLevel(lvl)
        """
        logger.setLevel(lvl)

    @property
    def level(self):
        """
        The level of data dumping that will occur
         - SUMMARY : A summary will be recorded
         - FULL : All possible dumping
         - CUSTOM : Some customised level of dumping
        When first set to CUSTOM this is equivalent to SUMMARY. It is then up
        to the user to specify what specifically is dumped
        """
        lvl = 'CUSTOM'
        if (self.dump_summary and not self.dump_any):
            lvl = 'SUMMARY'
        elif (self.dump_summary and self.dump_all):
            lvl = 'FULL'

        return lvl

    @level.setter
    def level(self, value):
        self._level = value
        self._apply_level()

    @property
    def dump_any(self):
        raise NotImplemented("This is an abstract class, "
                    "use subclass such as DynamicsDump or OptimDump")

    @property
    def dump_all(self):
        raise NotImplemented("This is an abstract class, "
                    "use subclass such as DynamicsDump or OptimDump")

    @property
    def dump_dir(self):
        if self._dump_dir is None:
            self.create_dump_dir()
        return self._dump_dir

    @dump_dir.setter
    def dump_dir(self, value):
        self._dump_dir = value
        if not self.create_dump_dir():
            self._dump_dir = None

    def create_dump_dir(self):
        """
        Checks dump directory exists, creates it if not
        """
        if self._dump_dir is None or len(self._dump_dir) == 0:
            self._dump_dir = DUMP_DIR

        dir_ok, self._dump_dir, msg = qtrlio.create_dir(
                    self._dump_dir, desc='dump')

        if not dir_ok:
            self.write_to_file = False
            msg += "\ndump file output will be suppressed."
            logger.error(msg)

        return dir_ok

    @property
    def fname_base(self):
        return self._fname_base

    @fname_base.setter
    def fname_base(self, value):
        if not _is_string(value):
            raise ValueError("File name base must be a string")
        self._fname_base = value
        self._summary_file_path = None

    @property
    def summary_file(self):
        if self._summary_file_path is None:
            fname = "{}-summary.{}".format(self._fname_base, self.dump_file_ext)
            self._summary_file_path = os.path.join(self.dump_dir, fname)
        return self._summary_file_path

    @summary_file.setter
    def summary_file(self, value):
        if not _is_string(value):
            raise ValueError("File path must be a string")
        self._summary_file_specified = True
        if os.path.abspath(value):
            self._summary_file_path = value
        elif '~' in value:
            self._summary_file_path = os.path.expanduser(value)
        else:
            self._summary_file_path = os.path.join(self.dump_dir, value)

class OptimDump(Dump):
    """
    A container for dumps of optimisation data generated during the pulse
    optimisation.

    Attributes
    ----------
    dump_summary : bool
        When True summary items are appended to the iter_summary

    iter_summary : list of :class:`optimizer.OptimIterSummary`
        Summary at each iteration

    dump_fid_err : bool
        When True values are appended to the fid_err_log

    fid_err_log : list of float
        Fidelity error at each call of the fid_err_func

    dump_grad_norm : bool
        When True values are appended to the fid_err_log

    grad_norm_log : list of float
        Gradient norm at each call of the grad_norm_log

    dump_grad : bool
        When True values are appended to the grad_log

    grad_log : list of ndarray
        Gradients at each call of the fid_grad_func

    """
    def __init__(self, optim, level='SUMMARY'):
        from qutip.control.optimizer import Optimizer
        if not isinstance(optim, Optimizer):
            raise TypeError("Must instantiate with {} type".format(
                                        Optimizer))
        self.parent = optim
        self._level = level
        self.reset()

    def reset(self):
        Dump.reset(self)
        self._apply_level()
        self.iter_summary = []
        self.fid_err_log = []
        self.grad_norm_log = []
        self.grad_log = []
        self._fname_base = 'optimdump'
        self._fid_err_file = None
        self._grad_norm_file = None

    def clear(self):
        del self.iter_summary[:]
        self.fid_err_log[:]
        self.grad_norm_log[:]
        self.grad_log[:]

    @property
    def dump_any(self):
        """True if anything other than the summary is to be dumped"""
        if (self.dump_fid_err or self.dump_grad_norm or self.dump_grad):
            return True
        else:
            return False

    @property
    def dump_all(self):
        """True if everything (ignoring the summary) is to be dumped"""
        if (self.dump_fid_err and self.dump_grad_norm and self.dump_grad):
            return True
        else:
            return False

    def _apply_level(self, level=None):
        if level is None:
            level = self._level

        if not _is_string(level):
            raise ValueError("Dump level must be a string")
        level = level.upper()
        if level == 'CUSTOM':
            if self._level == 'CUSTOM':
                # dumping level has not changed keep the same specific config
                pass
            else:
                # Switching to custom, start from SUMMARY
                level = 'SUMMARY'

        if level == 'SUMMARY':
            self.dump_summary = True
            self.dump_fid_err = False
            self.dump_grad_norm = False
            self.dump_grad = False
        elif level == 'FULL':
            self.dump_summary = True
            self.dump_fid_err = True
            self.dump_grad_norm = True
            self.dump_grad = True
        else:
            raise ValueError("No option for dumping level '{}'".format(level))

    def add_iter_summary(self):
        """add copy of current optimizer iteration summary"""
        optim = self.parent
        if optim.iter_summary is None:
            raise RuntimeError("Cannot add iter_summary as not available")
        ois = copy.copy(optim.iter_summary)
        ois.idx = len(self.iter_summary)
        self.iter_summary.append(ois)
        if self.write_to_file:
            if ois.idx == 0:
                f = open(self.summary_file, 'w')
                f.write("{}\n{}\n".format(
                            ois.get_header_line(self.summary_sep),
                            ois.get_value_line(self.summary_sep)))
            else:
                f = open(self.summary_file, 'a')
                f.write("{}\n".format(
                            ois.get_value_line(self.summary_sep)))

            f.close()
        return ois

    @property
    def fid_err_file(self):
        if self._fid_err_file is None:
            fname = "{}-fid_err_log.{}".format(self.fname_base,
                                                    self.dump_file_ext)
            self._fid_err_file = os.path.join(self.dump_dir, fname)
        return self._fid_err_file

    def update_fid_err_log(self, fid_err):
        """add an entry to the fid_err log"""
        self.fid_err_log.append(fid_err)
        if self.write_to_file:
            if len(self.fid_err_log) == 1:
                mode = 'w'
            else:
                mode = 'a'
            f = open(self.fid_err_file, mode)
            f.write("{}\n".format(fid_err))
            f.close()

    @property
    def grad_norm_file(self):
        if self._grad_norm_file is None:
            fname = "{}-grad_norm_log.{}".format(self.fname_base,
                                                    self.dump_file_ext)
            self._grad_norm_file = os.path.join(self.dump_dir, fname)
        return self._grad_norm_file

    def update_grad_norm_log(self, grad_norm):
        """add an entry to the grad_norm log"""
        self.grad_norm_log.append(grad_norm)
        if self.write_to_file:
            if len(self.grad_norm_log) == 1:
                mode = 'w'
            else:
                mode = 'a'
            f = open(self.grad_norm_file, mode)
            f.write("{}\n".format(grad_norm))
            f.close()

    def update_grad_log(self, grad):
        """add an entry to the grad log"""
        self.grad_log.append(grad)
        if self.write_to_file:
            fname = "{}-fid_err_gradients{}.{}".format(self.fname_base,
                                                        len(self.grad_log),
                                                        self.dump_file_ext)
            fpath = os.path.join(self.dump_dir, fname)
            np.savetxt(fpath, grad, delimiter=self.data_sep)

    def writeout(self, f=None):
        """write all the logs and the summary out to file(s)

        Parameters
        ----------
        f : filename or filehandle
            If specified then all summary and  object data will go in one file.
            If None is specified then type specific files will be generated
            in the dump_dir
            If a filehandle is specified then it must be a byte mode file
            as numpy.savetxt is used, and requires this.
        """
        fall = None
        # If specific file given then write everything to it
        if hasattr(f, 'write'):
            if not 'b' in f.mode:
                raise RuntimeError("File stream must be in binary mode")
            # write all to this stream
            fall = f
            fs = f
            closefall = False
            closefs = False
        elif f:
            # Assume f is a filename
            fall = open(f, 'wb')
            fs = fall
            closefs = False
            closefall = True
        else:
            self.create_dump_dir()
            closefall = False
            if self.dump_summary:
                fs = open(self.summary_file, 'wb')
                closefs = True

        if self.dump_summary:
            for ois in self.iter_summary:
                if ois.idx == 0:
                    fs.write(asbytes("{}\n{}\n".format(
                                ois.get_header_line(self.summary_sep),
                                ois.get_value_line(self.summary_sep))))
                else:
                    fs.write(asbytes("{}\n".format(
                                ois.get_value_line(self.summary_sep))))

            if closefs:
                fs.close()
                logger.info("Optim dump summary saved to {}".format(
                                                    self.summary_file))

        if self.dump_fid_err:
            if fall:
                fall.write(asbytes("Fidelity errors:\n"))
                np.savetxt(fall, self.fid_err_log)
            else:
                np.savetxt(self.fid_err_file, self.fid_err_log)

        if self.dump_grad_norm:
            if fall:
                fall.write(asbytes("gradients norms:\n"))
                np.savetxt(fall, self.grad_norm_log)
            else:
                np.savetxt(self.grad_norm_file, self.grad_norm_log)

        if self.dump_grad:
            g_num = 0
            for grad in self.grad_log:
                g_num += 1
                if fall:
                    fall.write(asbytes("gradients (call {}):\n".format(g_num)))
                    np.savetxt(fall, grad)
                else:
                    fname = "{}-fid_err_gradients{}.{}".format(self.fname_base,
                                                            g_num,
                                                            self.dump_file_ext)
                    fpath = os.path.join(self.dump_dir, fname)
                    np.savetxt(fpath, grad, delimiter=self.data_sep)

        if closefall:
            fall.close()
            logger.info("Optim dump saved to {}".format(f))
        else:
            if fall:
                logger.info("Optim dump saved to specified stream")
            else:
                logger.info("Optim dump saved to {}".format(self.dump_dir))

class DynamicsDump(Dump):
    """
    A container for dumps of dynamics data.
    Mainly time evolution calculations

    Attributes
    ----------
    dump_summary : bool
        If True a summary is recorded

    evo_summary : list of :class:`tslotcomp.EvoCompSummary'
        Summary items are appended if dump_summary is True
        at each recomputation of the evolution.

    dump_amps : bool
        If True control amplitudes are dumped

    dump_dyn_gen : bool
        If True the dynamics generators (Hamiltonians) are dumped

    dump_prop : bool
        If True propagators are dumped

    dump_prop_grad : bool
        If True propagator gradients are dumped

    dump_fwd_evo : bool
        If True forward evolution operators are dumped

    dump_onwd_evo : bool
        If True onward evolution operators are dumped

    dump_onto_evo : bool
        If True onto (or backward) evolution operators are dumped

    evo_dumps : list of :class:`EvoCompDumpItem`
        A new dump item is appended at each recomputation of the evolution.
        That is if any of the calculation objects are to be dumped.

    """
    def __init__(self, dynamics, level='SUMMARY'):
        from qutip.control.dynamics import Dynamics
        if not isinstance(dynamics, Dynamics):
            raise TypeError("Must instantiate with {} type".format(
                                        Dynamics))
        self.parent = dynamics
        self._level = level
        self.reset()

    def reset(self):
        Dump.reset(self)
        self._apply_level()
        self.evo_dumps = []
        self.evo_summary = []
        self._fname_base = 'dyndump'

    def clear(self):
        del self.evo_dumps[:]
        del self.evo_summary[:]

    @property
    def dump_any(self):
        """True if any of the calculation objects are to be dumped"""
        if (self.dump_amps or
                self.dump_dyn_gen or
                self.dump_prop or
                self.dump_prop_grad or
                self.dump_fwd_evo or
                self.dump_onwd_evo or
                self.dump_onto_evo):
            return True
        else:
            return False

    @property
    def dump_all(self):
        """True if all of the calculation objects are to be dumped"""
        dyn = self.parent
        if (self.dump_amps and
                    self.dump_dyn_gen and
                    self.dump_prop and
                    self.dump_prop_grad and
                    self.dump_fwd_evo and
                    (self.dump_onwd_evo) or
                    (self.dump_onwd_evo == dyn.fid_computer.uses_onwd_evo) and
                    (self.dump_onto_evo or
                    (self.dump_onto_evo == dyn.fid_computer.uses_onto_evo))):
            return True
        else:
            return False

    def _apply_level(self, level=None):
        dyn = self.parent
        if level is None:
            level = self._level

        if not _is_string(level):
            raise ValueError("Dump level must be a string")
        level = level.upper()
        if level == 'CUSTOM':
            if self._level == 'CUSTOM':
                # dumping level has not changed keep the same specific config
                pass
            else:
                # Switching to custom, start from SUMMARY
                level = 'SUMMARY'

        if level == 'SUMMARY':
            self.dump_summary = True
            self.dump_amps = False
            self.dump_dyn_gen = False
            self.dump_prop = False
            self.dump_prop_grad = False
            self.dump_fwd_evo = False
            self.dump_onwd_evo = False
            self.dump_onto_evo = False
        elif level == 'FULL':
            self.dump_summary = True
            self.dump_amps = True
            self.dump_dyn_gen = True
            self.dump_prop = True
            self.dump_prop_grad = True
            self.dump_fwd_evo = True
            self.dump_onwd_evo = dyn.fid_computer.uses_onwd_evo
            self.dump_onto_evo = dyn.fid_computer.uses_onto_evo
        else:
            raise ValueError("No option for dumping level '{}'".format(level))

    def add_evo_dump(self):
        """Add dump of current time evolution generating objects"""
        dyn = self.parent
        item = EvoCompDumpItem(self)
        item.idx = len(self.evo_dumps)
        self.evo_dumps.append(item)
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
        if self.dump_onto_evo:
            item.onto_evo = copy.deepcopy(dyn._onto_evo)

        if self.write_to_file:
            item.writeout()
        return item

    def add_evo_comp_summary(self, dump_item_idx=None):
        """add copy of current evo comp summary"""
        dyn = self.parent
        if dyn.tslot_computer.evo_comp_summary is None:
            raise RuntimeError("Cannot add evo_comp_summary as not available")
        ecs = copy.copy(dyn.tslot_computer.evo_comp_summary)
        ecs.idx = len(self.evo_summary)
        ecs.evo_dump_idx = dump_item_idx
        if dyn.stats:
            ecs.iter_num = dyn.stats.num_iter
            ecs.fid_func_call_num = dyn.stats.num_fidelity_func_calls
            ecs.grad_func_call_num = dyn.stats.num_grad_func_calls

        self.evo_summary.append(ecs)
        if self.write_to_file:
            if ecs.idx == 0:
                f = open(self.summary_file, 'w')
                f.write("{}\n{}\n".format(
                        ecs.get_header_line(self.summary_sep),
                        ecs.get_value_line(self.summary_sep)))
            else:
                f = open(self.summary_file, 'a')
                f.write("{}\n".format(ecs.get_value_line(self.summary_sep)))

            f.close()
        return ecs

    def writeout(self, f=None):
        """write all the dump items and the summary out to file(s)
        Parameters
        ----------
        f : filename or filehandle
            If specified then all summary and object data will go in one file.
            If None is specified then type specific files will be generated
            in the dump_dir
            If a filehandle is specified then it must be a byte mode file
            as numpy.savetxt is used, and requires this.
        """
        fall = None
        # If specific file given then write everything to it
        if hasattr(f, 'write'):
            if not 'b' in f.mode:
                raise RuntimeError("File stream must be in binary mode")
            # write all to this stream
            fall = f
            fs = f
            closefall = False
            closefs = False
        elif f:
            # Assume f is a filename
            fall = open(f, 'wb')
            fs = fall
            closefs = False
            closefall = True
        else:
            self.create_dump_dir()
            closefall = False
            if self.dump_summary:
                fs = open(self.summary_file, 'wb')
                closefs = True

        if self.dump_summary:
            for ecs in self.evo_summary:
                if ecs.idx == 0:
                    fs.write(asbytes("{}\n{}\n".format(
                            ecs.get_header_line(self.summary_sep),
                            ecs.get_value_line(self.summary_sep))))
                else:
                    fs.write(asbytes("{}\n".format(
                            ecs.get_value_line(self.summary_sep))))

            if closefs:
                fs.close()
                logger.info("Dynamics dump summary saved to {}".format(
                                                    self.summary_file))

        for di in self.evo_dumps:
            di.writeout(fall)

        if closefall:
            fall.close()
            logger.info("Dynamics dump saved to {}".format(f))
        else:
            if fall:
                logger.info("Dynamics dump saved to specified stream")
            else:
                logger.info("Dynamics dump saved to {}".format(self.dump_dir))

class DumpItem(object):
    """
    An item in a dump list
    """
    def __init__(self):
        pass

class EvoCompDumpItem(DumpItem):
    """
    A copy of all objects generated to calculate one time evolution
    Note the attributes are only set if the corresponding
    :class:`DynamicsDump` dump_ attribute is set.
    """
    def __init__(self, dump):
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
        """ write all the objects out to files

        Parameters
        ----------
        f : filename or filehandle
            If specified then all object data will go in one file.
            If None is specified then type specific files will be generated
            in the dump_dir
            If a filehandle is specified then it must be a byte mode file
            as numpy.savetxt is used, and requires this.
        """
        dump = self.parent
        fall = None
        closefall = True
        closef = False
        # If specific file given then write everything to it
        if hasattr(f, 'write'):
            if not 'b' in f.mode:
                raise RuntimeError("File stream must be in binary mode")
            # write all to this stream
            fall = f
            closefall = False
            f.write(asbytes("EVOLUTION COMPUTATION {}\n".format(self.idx)))
        elif f:
            fall = open(f, 'wb')
        else:
            # otherwise files for each type will be created
            fnbase = "{}-evo{}".format(dump._fname_base, self.idx)
            closefall = False

        #ctrl amps
        if not self.ctrl_amps is None:
            if fall:
                f = fall
                f.write(asbytes("Ctrl amps\n"))
            else:
                fname = "{}-ctrl_amps.{}".format(fnbase,
                                                dump.dump_file_ext)
                f = open(os.path.join(dump.dump_dir, fname), 'wb')
                closef = True
            np.savetxt(f, self.ctrl_amps, fmt='%14.6g',
                       delimiter=dump.data_sep)
            if closef: f.close()

        # dynamics generators
        if not self.dyn_gen is None:
            k = 0
            if fall:
                f = fall
                f.write(asbytes("Dynamics Generators\n"))
            else:
                fname = "{}-dyn_gen.{}".format(fnbase,
                                                dump.dump_file_ext)
                f = open(os.path.join(dump.dump_dir, fname), 'wb')
                closef = True
            for dg in self.dyn_gen:
                f.write(asbytes(
                        "dynamics generator for timeslot {}\n".format(k)))
                np.savetxt(f, self.dyn_gen[k], delimiter=dump.data_sep)
                k += 1
            if closef: f.close()

        # Propagators
        if not self.prop is None:
            k = 0
            if fall:
                f = fall
                f.write(asbytes("Propagators\n"))
            else:
                fname = "{}-prop.{}".format(fnbase,
                                                dump.dump_file_ext)
                f = open(os.path.join(dump.dump_dir, fname), 'wb')
                closef = True
            for dg in self.dyn_gen:
                f.write(asbytes("Propagator for timeslot {}\n".format(k)))
                np.savetxt(f, self.prop[k], delimiter=dump.data_sep)
                k += 1
            if closef: f.close()

        # Propagator gradient
        if not self.prop_grad is None:
            k = 0
            if fall:
                f = fall
                f.write(asbytes("Propagator gradients\n"))
            else:
                fname = "{}-prop_grad.{}".format(fnbase,
                                                dump.dump_file_ext)
                f = open(os.path.join(dump.dump_dir, fname), 'wb')
                closef = True
            for k in range(self.prop_grad.shape[0]):
                for j in range(self.prop_grad.shape[1]):
                    f.write(asbytes("Propagator gradient for timeslot {} "
                            "control {}\n".format(k, j)))
                    np.savetxt(f, self.prop_grad[k, j],
                               delimiter=dump.data_sep)
            if closef: f.close()

        # forward evolution
        if not self.fwd_evo is None:
            k = 0
            if fall:
                f = fall
                f.write(asbytes("Forward evolution\n"))
            else:
                fname = "{}-fwd_evo.{}".format(fnbase,
                                                dump.dump_file_ext)
                f = open(os.path.join(dump.dump_dir, fname), 'wb')
                closef = True
            for dg in self.dyn_gen:
                f.write(asbytes("Evolution from 0 to {}\n".format(k)))
                np.savetxt(f, self.fwd_evo[k], delimiter=dump.data_sep)
                k += 1
            if closef: f.close()

        # onward evolution
        if not self.onwd_evo is None:
            k = 0
            if fall:
                f = fall
                f.write(asbytes("Onward evolution\n"))
            else:
                fname = "{}-onwd_evo.{}".format(fnbase,
                                                dump.dump_file_ext)
                f = open(os.path.join(dump.dump_dir, fname), 'wb')
                closef = True
            for dg in self.dyn_gen:
                f.write(asbytes("Evolution from {} to end\n".format(k)))
                np.savetxt(f, self.fwd_evo[k], delimiter=dump.data_sep)
                k += 1
            if closef: f.close()

        # onto evolution
        if not self.onto_evo is None:
            k = 0
            if fall:
                f = fall
                f.write(asbytes("Onto evolution\n"))
            else:
                fname = "{}-onto_evo.{}".format(fnbase,
                                                dump.dump_file_ext)
                f = open(os.path.join(dump.dump_dir, fname), 'wb')
                closef = True
            for dg in self.dyn_gen:
                f.write(asbytes("Evolution from {} onto target\n".format(k)))
                np.savetxt(f, self.fwd_evo[k], delimiter=dump.data_sep)
                k += 1
            if closef: f.close()

        if closefall:
            fall.close()

class DumpSummaryItem(object):
    """A summary of the most recent iteration
    Abstract class only

    Attributes:
    idx : int
        Index in the summary list in which this is stored
    """
    min_col_width = 11
    summary_property_names = ()

    summary_property_fmt_type = ()

    summary_property_fmt_prec = ()

    @classmethod
    def get_header_line(cls, sep=' '):
        if sep == ' ':
            line = ''
            i = 0
            for a in cls.summary_property_names:
                if i > 0:
                    line += sep
                i += 1
                line += format(a, str(max(len(a), cls.min_col_width)) + 's')
        else:
            line = sep.join(cls.summary_property_names)
        return line

    def reset(self):
        self.idx = 0

    def get_value_line(self, sep=' '):
        line = ""
        i = 0
        for a in zip(self.summary_property_names,
                     self.summary_property_fmt_type,
                     self.summary_property_fmt_prec):
            if i > 0:
                line += sep
            i += 1
            v = getattr(self, a[0])
            w = max(len(a[0]), self.min_col_width)
            if v is not None:
                fmt = ''
                if sep == ' ':
                    fmt += str(w)
                else:
                    fmt += '0'
                if a[2] > 0:
                    fmt += '.' + str(a[2])
                fmt += a[1]
                line += format(v, fmt)
            else:
                if sep == ' ':
                    line += format('None', str(w) + 's')
                else:
                    line += 'None'

        return line