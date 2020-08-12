from ..optionclass import optionclass

__all__ = ["CoreOption"]

@optionclass("core")
class CoreOption:
    """
    Setting used by the Qobj.

    Value can be changed in qutip.settings.core

    Options
    -------

    auto_tidyup : bool
        use auto tidyup

    auto_tidyup_dims : boolTrue
        use auto tidyup dims on multiplication

    auto_herm : boolTrue
        detect hermiticity

    atol : float {1e-12}
        general absolute tolerance

    auto_tidyup_atol : float {1e-12}
        use auto tidyup absolute tolerance
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
        # use auto tidyup absolute tolerance
        "auto_tidyup_atol": 1e-12
    }
