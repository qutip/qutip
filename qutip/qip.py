"""Module replicating the qutip_qip package from within qutip."""
import sys

try:
    import qutip_qip

    sys.modules["qutip.qip"] = sys.modules["qutip_qip"]
    del qutip_qip
except ImportError:
    raise ImportError(
        "'qutip.qip' imports require the `qutip_qip` package. Install it with "
        "`pip install qutip-qip` or with `pip install qutip[qip]`."
    )
