from ..optionsclass import optionsclass

__all__ = ["CoreOptions"]

@optionsclass("core")
class CoreOptions:
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

    min_probability : float {1e-30}
        The minimun probability for measurement considered to be possible in
        :func:`measurement_statistics_povm`.
    """
    options = {
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
        # cutoff for impossible collapse in measurment
        "min_probability": 1e-30
    }
