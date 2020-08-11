from .optionclass import optionclass
import sys
import os
import logging


@optionclass("install")
class InstallSettings:
    try:
        _logger = logging.getLogger(__name__)
        _logger.addHandler(logging.NullHandler())
    except:
        _logger = None

    try:
        qutip_conf_dir = os.path.join(os.path.expanduser("~"), '.qutip')
        if not os.path.exists(qutip_conf_dir):
            os.mkdir(qutip_conf_dir)
        tmproot = os.path.join(qutip_conf_dir, 'coeffs')
        if not os.path.exists(tmproot):
            os.mkdir(tmproot)
        assert os.access(tmproot, os.W_OK)
        del qutip_conf_dir
    except Exception:
        tmproot = "."
    if tmproot not in sys.path:
        sys.path.insert(0, tmproot)

    options = {
        # debug mode for development
        "debug": False,
        # define whether log handler should be
        #   - default: switch based on IPython detection
        #   - stream: set up non-propagating StreamHandler
        #   - basic: call basicConfig
        #   - null: leave logging to the user
        "log_handler": 'default',
        # Allow for a colorblind mode that uses different colormaps
        # and plotting options by default.
        "colorblind_safe": False,
        # Note that since logging depends on settings,
        # if we want to do any logging here, it must be manually
        # configured, rather than through _logging.get_logger().
        "_logger": _logger,
        # Location of the saved string coefficients
        # Make sure it is in the "sys.path" if changing.
        "tmproot": tmproot
    }

    read_only_options = {
        # number of cpus (set at qutip import)
        "mkl_lib": None,
        # Flag if mkl_lib is found
        "has_mkl": False,
        # are we in IPython? Note that this cannot be
        # set by the RC file.
        "ipython": False
    }
