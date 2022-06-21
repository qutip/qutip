"""Module replicating the qutip_qip package from within qutip."""
import sys

try:
    import qutip_qip
    del qutip_qip
    sys.modules["qutip.qip"] = sys.modules["qutip_qip"]
except ImportError:
    raise ImportError(
        "Importing 'qutip.qip' requires the 'qutip_qip' package. Install it "
        "with `pip install qutip-qip` (for more details, go to "
        "https://qutip-qip.readthedocs.io/)."
    )
