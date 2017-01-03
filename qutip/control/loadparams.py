# -*- coding: utf-8 -*-
# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2014 and later, Alexander J G Pitchford
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

# @author: Alexander Pitchford
# @email1: agp1@aber.ac.uk
# @email2: alex.pitchford@gmail.com
# @organization: Aberystwyth University
# @supervisor: Daniel Burgarth

"""
Loads parameters for config, termconds, dynamics and Optimiser objects from a
parameter (ini) file with appropriate sections and options, these being
Sections: optimconfig, termconds, dynamics, optimizer
The options are assumed to be properties for these classes
Note that new attributes will be created, even if they are not usually
defined for that object
"""

import numpy as np

try:
    # Python 3
    from configparser import ConfigParser
except:
    # Python 2
    from ConfigParser import SafeConfigParser

# QuTiP logging
from qutip import Qobj

import qutip.logging_utils as logging
logger = logging.get_logger()

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

def load_parameters(file_name, config=None, term_conds=None,
                    dynamics=None, optim=None, pulsegen=None,
                    obj=None, section=None):
    """
    Import parameters for the optimisation objects
    Will throw a ValueError if file_name does not exist
    """
    try:
        parser = SafeConfigParser()
    except:
        parser = ConfigParser()

    readFiles = parser.read(str(file_name))
    if len(readFiles) == 0:
        raise ValueError("Parameter file '{}' not found".format(file_name))

    if config is not None:
        s = 'optimconfig'
        try:
            attr_names = parser.options(s)
            for a in attr_names:
                set_param(parser, s, a, config, a)
        except Exception as e:
            logger.warn("Unable to load {} parameters:({}) {}".format(
                s, type(e).__name__, e))

    if term_conds is not None:
        s = 'termconds'
        try:
            attr_names = parser.options(s)
            for a in attr_names:
                set_param(parser, s, a, term_conds, a)
        except Exception as e:
            logger.warn("Unable to load {} parameters:({}) {}".format(
                s, type(e).__name__, e))

    if dynamics is not None:
        s = 'dynamics'
        try:
            attr_names = parser.options(s)
            for a in attr_names:
                set_param(parser, s, a, dynamics, a)
        except Exception as e:
            logger.warn("Unable to load {} parameters:({}) {}".format(
                s, type(e).__name__, e))

    if optim is not None:
        s = 'optimizer'
        try:
            attr_names = parser.options(s)
            for a in attr_names:
                set_param(parser, s, a, optim, a)
        except Exception as e:
            logger.warn("Unable to load {} parameters:({}) {}".format(
                s, type(e).__name__, e))

    if pulsegen is not None:
        s = 'pulsegen'
        try:
            attr_names = parser.options(s)
            for a in attr_names:
                set_param(parser, s, a, pulsegen, a)
        except Exception as e:
            logger.warn("Unable to load {} parameters:({}) {}".format(
                s, type(e).__name__, e))

    if obj is not None:
        if not _is_string(section):
            raise ValueError("Section name must be given when loading "
                "parameters of general object")
        s = section
        try:
            attr_names = parser.options(s)
            for a in attr_names:
                set_param(parser, s, a, obj, a)
        except Exception as e:
            logger.warn("Unable to load {} parameters:({}) {}".format(
                s, type(e).__name__, e))

def set_param(parser, section, option, obj, attrib_name):
    """
    Set the object attribute value based on the option value from the
    config file.
    If the attribute exists already, then its datatype
    is used to call the appropriate parser.get method
    Otherwise the parameter is assumed to be a string

    """
    val = parser.get(section, attrib_name)

    dtype = None
    if hasattr(obj, attrib_name):
        a = getattr(obj, attrib_name)
        dtype = type(a)
    else:
        logger.warn("Unable to load parameter {}.{}\n"
                    "Attribute does not exist".format(section, attrib_name))
        return

    if isinstance(a, Qobj):
        try:
            q = Qobj(eval(val))
        except:
            raise ValueError("Value '{}' cannot be used to generate a Qobj"
                             " in parameter file [{}].{}".format(
                                 val, section, option))
        setattr(obj, attrib_name, q)
    elif isinstance(a, np.ndarray):
        try:
            arr = np.array(eval(val), dtype=a.dtype)
        except:
            raise ValueError("Value '{}' cannot be used to generate an ndarray"
                             " in parameter file [{}].{}".format(
                                 val, section, option))
        setattr(obj, attrib_name, arr)
    elif isinstance(a, list):
        try:
            l = list(eval(val))
        except:
            raise ValueError("Value '{}' cannot be used to generate a list"
                             " in parameter file [{}].{}".format(
                                 val, section, option))
        setattr(obj, attrib_name, l)
    elif dtype == float:
        try:
            f = parser.getfloat(section, attrib_name)
        except:
            try:
                f = eval(val)
            except:
                raise ValueError(
                    "Value '{}' cannot be cast or evaluated as a "
                    "float in parameter file [{}].{}".format(
                        val, section, option))
        setattr(obj, attrib_name, f)
    elif dtype == complex:
        try:
            c = complex(val)
        except:
            raise ValueError("Value '{}' cannot be cast as complex"
                             " in parameter file [{}].{}".format(
                                 val, section, option))
        setattr(obj, attrib_name, c)
    elif dtype == int:
        try:
            i = parser.getint(section, attrib_name)
        except:
            raise ValueError("Value '{}' cannot be cast as an int"
                             " in parameter file [{}].{}".format(
                                 val, section, option))
        setattr(obj, attrib_name, i)
    elif dtype == bool:
        try:
            b = parser.getboolean(section, attrib_name)
        except:
            raise ValueError("Value '{}' cannot be cast as a bool"
                             " in parameter file [{}].{}".format(
                                 val, section, option))
        setattr(obj, attrib_name, b)
    else:
        try:
            val = parser.getfloat(section, attrib_name)
        except:
            try:
                val = parser.getboolean(section, attrib_name)
            except:
                pass

        setattr(obj, attrib_name, val)
