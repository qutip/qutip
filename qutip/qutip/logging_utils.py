# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2014 and later, Paul D. Nation and Robert J. Johansson.
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
This module contains internal-use functions for configuring and writing to
debug logs, using Python's internal logging functionality by default.
"""

# IMPORTS
from __future__ import absolute_import
import inspect
import logging

import qutip.settings

# EXPORTS
NOTSET = logging.NOTSET
DEBUG_INTENSE = logging.DEBUG - 4
DEBUG_VERBOSE = logging.DEBUG - 2
DEBUG = logging.DEBUG
INFO = logging.INFO
WARN = logging.WARN
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

__all__ = ['get_logger']

# META-LOGGING

metalogger = logging.getLogger(__name__)
metalogger.addHandler(logging.NullHandler())


# FUNCTIONS

def get_logger(name=None):
    """
    Returns a Python logging object with handlers configured
    in accordance with ~/.qutiprc. By default, this will do
    something sensible to integrate with IPython when running
    in that environment, and will print to stdout otherwise.

    Note that this function uses a bit of magic, and thus should
    not be considered part of the QuTiP API. Rather, this function
    is for internal use only.

    Parameters
    ----------

    name : str
        Name of the logger to be created. If not passed,
        the name will automatically be set to the name of the
        calling module.
    """
    if name is None:
        try:
            calling_frame = inspect.stack()[1][0]
            calling_module = inspect.getmodule(calling_frame)
            name = (calling_module.__name__
                    if calling_module is not None else '<none>')

        except Exception:
            metalogger.warn('Error creating logger.', exc_info=1)
            name = '<unknown>'

    logger = logging.getLogger(name)

    policy = qutip.settings.log_handler

    if policy == 'default':
        # Let's try to see if we're in IPython mode.
        policy = 'basic' if qutip.settings.ipython else 'stream'

    metalogger.debug("Creating logger for {} with policy {}.".format(
        name, policy
    ))

    if policy == 'basic':
        # Add no handlers, just let basicConfig do it all.
        # This is nice for working with IPython, since
        # it will use its own handlers instead of our StreamHandler
        # below.
        if qutip.settings.debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig()

    elif policy == 'stream':
        formatter = logging.Formatter(
            '[%(asctime)s] %(name)s[%(process)s]: '
            '%(funcName)s: %(levelname)s: %(message)s',
            '%Y-%m-%d %H:%M:%S')
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # We're handling things here, so no propagation out.
        logger.propagate = False

    elif policy == 'null':
        # We need to add a NullHandler so that debugging works
        # at all, but this policy leaves it to the user to
        # make their own handlers. This is particularly useful
        # for capturing to logfiles.
        logger.addHandler(logging.NullHandler())

    if qutip.settings.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARN)

    return logger
