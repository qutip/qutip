# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 14:46:26 2014

@author: Alexander Pitchford
@email1: agp1@aber.ac.uk
@email2: alex.pitchford@gmail.com
@organization: Aberystwyth University
@supervisor: Daniel Burgarth

The code in this file was is intended for use in not-for-profit research,
teaching, and learning. Any other applications may require additional
licensing

Pulse generator - Generate pulses for the timeslots
Each class defines a gen_pulse function that produces a float array of
size num_tslots. Each class produces a differ type of pulse.
See the class and gen_pulse function descriptions for details
"""

import numpy as np
import dynamics as dynamics
import errors as errors

def create_pulse_gen(pulse_type='RND', dyn=None):
    """
    Create and return a pulse generator object matching the given type.
    The pulse generators each produce a different type of pulse,
    see the gen_pulse function description for details.
    These are the non-periodic options:
        RND - Random value in each timeslot
        LIN - Linear, i.e. contant gradient over the time
        ZERO - special case of the LIN pulse, where the gradient is 0
    These are the periodic options
        SINE - Sine wave
        SQUARE - Square wave
        SAW - Saw tooth wave
        TRIANGLE - Triangular wave
    If a Dynamics object is passed in then this is used in instantiate
    the PulseGen, meaning that some timeslot and amplitude properties
    are copied over.
    """
    
    if (pulse_type == 'RND'):
        return PulseGen_Random(dyn)
    elif (pulse_type == 'LIN'):
        return PulseGen_Linear(dyn)
    elif (pulse_type == 'ZERO'):
        return PulseGen_Zero(dyn)
    elif (pulse_type == 'SINE'):
        return PulseGen_Sine(dyn)
    elif (pulse_type == 'SQUARE'):
        return PulseGen_Square(dyn)
    elif (pulse_type == 'SAW'):
        return PulseGen_Saw(dyn)
    elif (pulse_type == 'TRIANGLE'):
        return PulseGen_Triangle(dyn)

# [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
class PulseGen:
    """
    Pulse generator
    Base class for all Pulse generators
    The object can optionally be instantiated with a Dynamics object,
    in which case the time slots and amplitude scaling are copied from that.
    Otherwise the class can be used independently by setting:
    tau (array of timeslot durations) 
    or 
    num_tslots and pulse_time for equally spaced timeslots
    """
    def __init__(self, dyn=None):
        self.parent = dyn
        self.reset()
        
    def reset(self):
        """
        reset any configuration data and
        clear any temporarily held status data
        """
        if (isinstance(self.parent, dynamics.Dynamics)):
            dyn = self.parent
            self.msg_level = dyn.msg_level
            self.num_tslots = dyn.num_tslots
            self.pulse_time = dyn.evo_time
            self.scaling = dyn.initial_ctrl_scaling
            self.offset = dyn.initial_ctrl_offset
            self.tau = dyn.tau
        else:
            self.msg_level = 0
            self.num_tslots = 100
            self.pulse_time = 1.0
            self.scaling = 1.0
            self.tau = None
            self.offset = 0.0
        
        self._pulse_initialised = False
        self.periodic = False
        
    def gen_pulse(self):
        """
        returns the pulse as an array of vales for each timeslot
        Must be implemented by subclass
        """
        # must be implemented by subclass
        f = self.__class__.__name__ + ".get_fid_err"
        m = "No method defined for getting fidelity error." + \
                " Suspect base class was used where sub class should have been"
        raise errors.UsageError(funcname=f, msg=m)
        
    def init_pulse(self):
        """
        Initialise the pulse parameters
        """
        if (self.tau == None):
            self.tau =  np.ones(self.num_tslots, dtype = 'f') * \
                            self.pulse_time/self.num_tslots
        self._pulse_initialised = True
# ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]

class PulseGen_Zero(PulseGen):
    """
    Generates a flat pulse
    """
    def gen_pulse(self):
        """
        Generate a pulse with the same value in every timeslot.
        The value will be zero, unless the offset is not zero,
        in which case it will be the offset
        """
        pulse = np.zeros(self.num_tslots) + self.offset
        return pulse
        
class PulseGen_Random(PulseGen):
    """
    Generates random pulses
    """
    def gen_pulse(self):
        """
        Generate a pulse of random values between 1 and -1
        Values are scaled using the scaling property
        and shifted using the offset property
        Returns the pulse as an array of vales for each timeslot
        """
        pulse = (2*np.random.random(self.num_tslots) - 1)* \
                        self.scaling + self.offset
        return pulse

class PulseGen_Linear(PulseGen):
    """
    Generates linear pulses
    """
    def reset(self):
        """
        reset any configuration data and
        """
        PulseGen.reset(self)

        self.gradient = None
        self.start_val = -1.0
        self.end_val = 1.0
    
    def init_pulse(self, gradient=None, start_val=None, end_val=None):
        """
        Calulate the gradient if pulse is defined by start and end point values
        """
        PulseGen.init_pulse(self)
        if (start_val is not None and end_val is not None):
            self.start_val = start_val
            self.end_val = end_val
        if (self.start_val is not None and self.end_val is not None):
            self.gradient = float(self.end_val - self.start_val) / \
                            (self.pulse_time - self.tau[-1])
        
    def gen_pulse(self, gradient=None, start_val=None, end_val=None):
        """
        Generate a linear pulse using either the gradient and start value
        or using the end point to calulate the gradient
        Note that the scaling and offset parameters are still applied,
        so unless these values are the default 1.0 and 0.0, then the
        actual gradient etc will be different
        Returns the pulse as an array of vales for each timeslot
        """
        if (gradient is not None or \
                start_val is not None or end_val is not None):
            self.init_pulse(gradient, start_val, end_val)
        if (not self._pulse_initialised):
            self.init_pulse()
            
        pulse = np.empty(self.num_tslots)
        t = 0.0
        for k in range(self.num_tslots):
            y = self.gradient*t + self.start_val
            pulse[k] = self.scaling*y + self.offset
            t = t + self.tau[k]
        return pulse
        
# [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
class PulseGen_Periodic(PulseGen):
    """
    Intermediate class for all periodic pulse generators
    All of the periodic pulses range from -1 to 1
    All have a start phase that can be set between 0 and 2pi
    """
    def reset(self):
        """
        reset any configuration data and
        """
        PulseGen.reset(self)
        self.periodic = True
        self.num_waves = None
        self.freq = 1.0
        self.wavelen = None
        self.start_phase = 0.0
        
    def init_pulse(self, num_waves=None, wavelen=None, \
                    freq=None, start_phase=None):
        """
        Calculate the wavelength, frequency, number of waves etc 
        from the each other and the other parameters
        If num_waves is given then the other parameters are worked from this
        Otherwise if the wavelength is given then it is the driver
        Otherwise the frequency is used to calculate wavelength and num_waves
        """
        PulseGen.init_pulse(self)
        
        if (start_phase is not None):
            self.start_phase = start_phase
            
        if (num_waves is not None or wavelen is not None or freq is not None):
            self.num_waves = num_waves
            self.wavelen = wavelen
            self.freq = freq
            
        if (self.num_waves is not None):
            self.freq = float(self.num_waves) / self.pulse_time
            self.wavelen = 1.0/self.freq
        elif (self.wavelen is not None):
            self.freq = 1.0/self.wavelen
            self.num_waves = self.wavelen*self.pulse_time
        else:
            self.wavelen = 1.0/self.freq
            self.num_waves = self.wavelen*self.pulse_time
            
# ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
        
# [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
class PulseGen_Sine(PulseGen_Periodic):
    """
    Generates sine wave pulses
    """
    def gen_pulse(self, num_waves=None, wavelen=None, \
                    freq=None, start_phase=None):
        """
        Generate a sine wave pulse
        If no params are provided then the class object attributes are used.
        If they are provided, then these will reinitialise the object attribs.
        returns the pulse as an array of vales for each timeslot
        """
        if (start_phase is not None):
            self.start_phase = start_phase
            
        if (num_waves is not None or wavelen is not None or freq is not None):
            self.init_pulse(num_waves, wavelen, freq, start_phase)
            
        if (not self._pulse_initialised):
            self.init_pulse()
        
        pulse = np.empty(self.num_tslots)
        t = 0.0
        for k in range(self.num_tslots):
            phase = 2*np.pi*self.freq*t + self.start_phase
            pulse[k] = self.scaling*np.sin(phase) + self.offset
            t = t + self.tau[k]
        return pulse
        
# ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
    
# [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
class PulseGen_Square(PulseGen_Periodic):
    """
    Generates square wave pulses
    """
    def gen_pulse(self, num_waves=None, wavelen=None, \
                    freq=None, start_phase=None):
        """
        Generate a square wave pulse
        If no parameters are pavided then the class object attributes are used.
        If they are provided, then these will reinitialise the object attribs
        """
        if (start_phase is not None):
            self.start_phase = start_phase
            
        if (num_waves is not None or wavelen is not None or freq is not None):
            self.init_pulse(num_waves, wavelen, freq, start_phase)
            
        if (not self._pulse_initialised):
            self.init_pulse()
        
        pulse = np.empty(self.num_tslots)
        t = 0.0
        for k in range(self.num_tslots):
            phase = 2*np.pi*self.freq*t + self.start_phase
            x = phase/(2*np.pi)
            y = 4*np.floor(x) - 2*np.floor(2*x) + 1
            pulse[k] = self.scaling*y + self.offset
            t = t + self.tau[k]
        return pulse
        
# ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
        
# [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
class PulseGen_Saw(PulseGen_Periodic):
    """
    Generates saw tooth wave pulses
    """
    def gen_pulse(self, num_waves=None, wavelen=None, \
                    freq=None, start_phase=None):
        """
        Generate a saw tooth wave pulse
        If no parameters are pavided then the class object attributes are used.
        If they are provided, then these will reinitialise the object attribs
        """
        if (start_phase is not None):
            self.start_phase = start_phase
            
        if (num_waves is not None or wavelen is not None or freq is not None):
            self.init_pulse(num_waves, wavelen, freq, start_phase)
            
        if (not self._pulse_initialised):
            self.init_pulse()
        
        pulse = np.empty(self.num_tslots)
        t = 0.0
        for k in range(self.num_tslots):
            phase = 2*np.pi*self.freq*t + self.start_phase
            x = phase/(2*np.pi)
            y = 2*(x - np.floor(0.5 + x))
            pulse[k] = self.scaling*y + self.offset
            t = t + self.tau[k]
        return pulse
        
# ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]
        
# [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[
class PulseGen_Triangle(PulseGen_Periodic):
    """
    Generates triangular wave pulses
    """
    def gen_pulse(self, num_waves=None, wavelen=None, \
                    freq=None, start_phase=None):
        """
        Generate a sine wave pulse
        If no parameters are pavided then the class object attributes are used.
        If they are provided, then these will reinitialise the object attribs
        """
        if (start_phase is not None):
            self.start_phase = start_phase
            
        if (num_waves is not None or wavelen is not None or freq is not None):
            self.init_pulse(num_waves, wavelen, freq, start_phase)
            
        if (not self._pulse_initialised):
            self.init_pulse()
        
        pulse = np.empty(self.num_tslots)
        t = 0.0
        for k in range(self.num_tslots):
            phase = 2*np.pi*self.freq*t + self.start_phase
            x = phase/(2*np.pi)
            y = 2*np.abs(2*(x - np.floor(0.5 + x))) - 1
            pulse[k] = self.scaling*y + self.offset
            t = t + self.tau[k]
            
        return pulse
        
# ]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]