"""
This module contains internal-use functions for configuring and writing to
debug logs, using Python's internal logging functionality by default.
"""

# IMPORTS
from __future__ import absolute_import
import inspect
import logging

from qutip.settings import settings

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

    policy = settings.log_handler

    if policy == 'default':
        # Let's try to see if we're in IPython mode.
        policy = 'basic' if settings.ipython else 'stream'

    metalogger.debug("Creating logger for {} with policy {}.".format(
        name, policy
    ))

    if policy == 'basic':
        # Add no handlers, just let basicConfig do it all.
        # This is nice for working with IPython, since
        # it will use its own handlers instead of our StreamHandler
        # below.
        if settings.debug:
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

    if settings.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARN)

    return logger
