"""Module replicating the qutip_qtrl package from within qutip."""
import sys

try:
    import qutip_qtrl
    del qutip_qtrl
    sys.modules["qutip.control"] = sys.modules["qutip_qtrl"]
except ImportError:
    raise ImportError(
        "Importing 'qutip.control' requires the 'qutip_qtrl' package. "
        "Install it with `pip install qutip-qtrl` (for more details, go to "
        "https://qutip-qtrl.readthedocs.io/)."
    )
