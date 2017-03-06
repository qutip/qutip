# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, QuSTaR.
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
import os
import qutip
import qutip.settings as qset
import warnings
try:
    import ConfigParser as configparser #py27
except:
    import configparser #py3x

def has_qutip_rc():
    """
    Checks to see if the qutiprc file exists in the default
    location, i.e. HOME/.qutip/qutiprc
    """
    qutip_conf_dir = os.path.join(os.path.expanduser("~"), '.qutip')
    if os.path.exists(qutip_conf_dir):
        qutip_rc_file = os.path.join(qutip_conf_dir,'qutiprc')
        qrc_exists = os.path.isfile(qutip_rc_file) 
        if qrc_exists:
            return True, qutip_rc_file
        else:
            return False, ''
    else:
        return False, ''


def generate_qutiprc():
    """
    Generate a blank qutiprc file.
    """
    # Check for write access to home dir
    if not os.access(os.path.expanduser("~"), os.W_OK):
        return False
    qutip_conf_dir = os.path.join(os.path.expanduser("~"), '.qutip')
    if not os.path.exists(qutip_conf_dir):
        try:
            os.mkdir(qutip_conf_dir)
        except:
            warnings.warn('Cannot write config file to user home dir.')
            return False
    qutip_rc_file = os.path.join(qutip_conf_dir,'qutiprc')
    qrc_exists = os.path.isfile(qutip_rc_file)
    if qrc_exists:
        #Do not overwrite
        return False
    else:
        #Write a basic file with qutip section
        cfgfile = open(qutip_rc_file,'w')
        config = configparser.ConfigParser()
        config.add_section('qutip')
        config.write(cfgfile)
        cfgfile.close()
        return True 
        

def load_rc_config(rc_file):
    """
    Loads the configuration data from the
    qutiprc file
    """
    config = configparser.ConfigParser()
    _valid_keys ={'auto_tidyup' : config.getboolean, 'auto_herm' : config.getboolean, 
            'atol': config.getfloat, 'auto_tidyup_atol' : config.getfloat,
            'num_cpus' : config.getint, 'debug' : config.getboolean, 
            'log_handler' : config.getboolean, 'colorblind_safe' : config.getboolean,
            'openmp_thresh': config.getint}
    config.read(rc_file)
    if config.has_section('qutip'):
        opts = config.options('qutip')
        for op in opts:
            if op in _valid_keys.keys():
                setattr(qset, op, _valid_keys[op]('qutip',op))
            else:
                raise Exception('Invalid config variable in qutiprc.')
    else:
        raise configparser.NoSectionError('qutip')
        
    if config.has_section('compiler'):
        _valid_keys = ['CC', 'CXX']
        opts = config.options('compiler')
        for op in opts:
            up_op = op.upper()
            if up_op in _valid_keys:
                os.environ[up_op] = config.get('compiler', op)
            else:
                raise Exception('Invalid config variable in qutiprc.')
 
 
def has_rc_key(rc_file, key):
    config = configparser.ConfigParser()
    config.read(rc_file)
    if config.has_section('qutip'):
        opts = config.options('qutip')
        if key in opts:
            return True
        else:
            return False
    else:
        raise configparser.NoSectionError('qutip')
       
        
def write_rc_key(rc_file, key, value):
    """
    Writes a single key value to the qutiprc file
    
    Parameters
    ----------
    rc_file : str
        String specifying file location.
    key : str
        The key name to be written.
    value : int/float/bool
        Value corresponding to given key.
    """
    if not os.access(os.path.expanduser("~"), os.W_OK):
        return
    cfgfile = open(rc_file,'w')
    config = configparser.ConfigParser()
    if not config.has_section('qutip'):
        config.add_section('qutip')
    config.set('qutip',key,str(value))
    config.write(cfgfile)
    cfgfile.close()
    