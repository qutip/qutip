# -*- coding: utf-8 -*-
"""
Created on Tue May 06 15:03:42 2014

@author: Alexander Pitchford
@email1: agp1@aber.ac.uk
@email2: alex.pitchford@gmail.com
@organization: Aberystwyth University
@supervisor: Daniel Burgarth

The code in this file was is intended for use in not-for-profit research,
teaching, and learning. Any other applications may require additional
licensing
Loads parameters for config, termConds, dynamics and Optimiser objects from a
parameter (ini) file with approxiate sections and options, these being
Sections: optimconfig, termconds, dynamics, optimizer
The options are assumed to be properties for these classes
Note that new attributes will be created, even if they are not usually
defined for that object
"""

from ConfigParser import SafeConfigParser

def load_parameters(file_name, config=None, term_conds=None, \
                        dynamics=None, optim=None):
    """
    Import parameters for the optimisation objects
    Will throw a ValueError if file_name does not exist
    """
    parser = SafeConfigParser()
    readFiles = parser.read(file_name)
    if (len(readFiles) == 0):
        raise ValueError("Parameter file '{}' not found".format(file_name))
    
    if (config != None):
        s = 'optimconfig'
        attrNames = parser.options(s)
        for a in attrNames:
            set_param(parser, s, a, config, a)
            
    if (term_conds != None):
        s = 'termconds'
        attrNames = parser.options(s)
        for a in attrNames:
            set_param(parser, s, a, term_conds, a)
            
    if (dynamics != None):
        s = 'dynamics'
        attrNames = parser.options(s)
        for a in attrNames:
            set_param(parser, s, a, dynamics, a)
            
    if (optim != None):
        s = 'optimizer'
        attrNames = parser.options(s)
        for a in attrNames:
            set_param(parser, s, a, optim, a)
            
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
    if (hasattr(obj, attrib_name)):
        a = getattr(obj, attrib_name)
        dtype = type(a)
                
    if (dtype == float):
        try:
            f = parser.getfloat(section, attrib_name)
        except:
            try:
                f = eval(val)
            except:
                raise ValueError("Value '{}' cannot be cast or evaluated as a "
                            "float in parameter file [{}].{}".format(val, \
                            section, option))
        setattr(obj, attrib_name, f)
    elif (dtype == complex):
        try:
            c = complex(val)
        except:
            raise ValueError("Value '{}' cannot be cast as complex" 
                        " in parameter file [{}].{}".format(val, \
                        section, option))
        setattr(obj, attrib_name, c)          
    elif (dtype == int):
        try:
            i = parser.getint(section, attrib_name)
        except:
            raise ValueError("Value '{}' cannot be cast as an int"
                        " in parameter file [{}].{}".format(val, \
                        section, option))
        setattr(obj, attrib_name, i)
    elif (dtype == bool):
        try:
            b = parser.getboolean(section, attrib_name)
        except:
            raise ValueError("Value '{}' cannot be cast as a bool"
                        " in parameter file [{}].{}".format(val, \
                        section, option))
        setattr(obj, attrib_name, b)
    else:
        setattr(obj, attrib_name, val)