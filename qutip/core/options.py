from ..optionsclass import QutipOptions
from ..settings import settings

__all__ = ["CoreOptions"]


class CoreOptions(QutipOptions):
    """
    Settings used by the Qobj.  Values can be changed in qutip.settings.core.

    Options
    -------
    auto_tidyup : bool
        Whether to tidyup during sparse operations.

    auto_tidyup_dims : boolTrue
        use auto tidyup dims on multiplication

    auto_herm : boolTrue
        detect hermiticity

    atol : float {1e-12}
        general absolute tolerance

    rtol : float {1e-12}
        general relative tolerance
        Used to choose QobjEvo.expect output type

    auto_tidyup_atol : float {1e-14}
        The absolute tolerance used in automatic tidyup (see the ``auto_tidyup``
        parameter above) and the default value of ``atol`` used in
        :method:`Qobj.tidyup`.

    function_coefficient_style : str {"auto"}
        The signature expected by function coefficients. The options are:

        - "pythonic": the signature should be ``f(t, ...)`` where ``t``
          is the time and the ``...`` are the remaining arguments passed
          directly into the function. E.g. ``f(t, w, b=5)``.

        - "dict": the signature shoule be ``f(t, args)`` where ``t`` is
          the time and ``args`` is a dict containing the remaining arguments.
          E.g. ``f(t, {"w": w, "b": 5})``.

        - "auto": select automatically between the two options above based
          on the signature of the supplied function. If the function signature
          is exactly ``f(t, args)`` then ``dict`` is used. Otherwise
          ``pythonic`` is used.
    """
    default = {
        # use auto tidyup
        "auto_tidyup": True,
        # use auto tidyup dims on multiplication
        "auto_tidyup_dims": True,
        # detect hermiticity
        "auto_herm": True,
        # general absolute tolerance
        "atol": 1e-12,
        # general relative tolerance
        "rtol": 1e-12,
        # use auto tidyup absolute tolerance
        "auto_tidyup_atol": 1e-14,
        # signature style expected by function coefficients
        "function_coefficient_style": "auto",
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
    }


settings.core = CoreOptions
