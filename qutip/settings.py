# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
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
This module contains settings for the QuTiP graphics, multiprocessing, and
tidyup functionality, etc.
"""
from __future__ import absolute_import
# use auto tidyup
auto_tidyup = True
# detect hermiticity
auto_herm = True
# general absolute tolerance
atol = 1e-12
# use auto tidyup absolute tolerance
auto_tidyup_atol = 1e-12
# number of cpus (set at qutip import)
num_cpus = 1
# flag indicating if fortran module is installed
fortran = False
# flag indicating if scikits.umfpack is installed
umfpack = False
# debug mode for development
debug = False
# are we in IPython? Note that this cannot be
# set by the RC file.
ipython = False
# define whether log handler should be
#   - default: switch based on IPython detection
#   - stream: set up non-propagating StreamHandler
#   - basic: call basicConfig
#   - null: leave logging to the user
log_handler = 'default'


# Note that since logging depends on settings,
# if we want to do any logging here, it must be manually
# configured, rather than through _logging.get_logger().
import logging
_logger = logging.getLogger(__name__)
_logger.addHandler(logging.NullHandler())
del logging # Don't leak names!

# Try to pull in configobj to do nicer handling of
# config files instead of doing manual parsing.
try:
    import configobj as _cobj
except ImportError:
    _logger.warn("configobj missing.", exc_info=1)
    _cobj = None


def load_rc_file(rc_file):
    """
    Load settings for the qutip RC file, by default .qutiprc in the user's home
    directory.
    """
    global auto_tidyup, auto_herm, auto_tidyup_atol, num_cpus, debug, atol
    global log_handler

    if _cobj is None:
        raise ImportError("Config file parsing requires configobj. Skipping.")

    import pkg_resources
    import validate
    config = _cobj.ConfigObj(rc_file,
        configspec=pkg_resources.resource_filename('qutip', 'configspec.ini')
    )
    validator = validate.Validator()
    result = config.validate(validator)

    if result != True: # this is un-Pythonic, but how the result gets
                       # returned...
        # OK, find which keys are bad.
        bad_keys = {key for key, val in result.iteritems() if not val}
        _logger.warn('Invalid configuration options in {}: {}'.format(
            rc_file, bad_keys
        ))
    else:
        bad_keys = {}

    for config_key in (
        'auto_tidyup', 'auto_herm', 'atol', 'auto_tidyup_atol',
        'num_cpus', 'debug', 'log_handler'
    ):
        if config_key in config and config_key not in bad_keys:
            _logger.debug(
                "Applying configuration setting {} = {}.".format(
                    config_key, config[config_key]
                )
            )
            globals()[config_key] = config[config_key]
