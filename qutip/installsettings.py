from .optionclass import optionclass

@optionclass("install")
class InstallSettings:
    try:
        import logging
        _logger = logging.getLogger(__name__)
        _logger.addHandler(logging.NullHandler())
        del logging  # Don't leak names!
    except:
        _logger = None

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
        "_logger": _logger
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
