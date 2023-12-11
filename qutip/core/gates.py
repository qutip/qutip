from itertools import product
from functools import partial, reduce
from operator import mul

import numpy as np
import scipy.sparse as sp
from . import Qobj, qeye, sigmax, fock_dm, qdiags, qeye_like


__all__ = [
    "rx",
    "ry",
    "rz",
    "sqrtnot",
    "snot",
    "phasegate",
    "qrot",
    "cy_gate",
    "cz_gate",
    "s_gate",
    "t_gate",
    "cs_gate",
    "ct_gate",
    "cphase",
    "cnot",
    "csign",
    "berkeley",
    "swapalpha",
    "swap",
    "iswap",
    "sqrtswap",
    "sqrtiswap",
    "fredkin",
    "molmer_sorensen",
    "toffoli",
    "hadamard_transform",
    "qubit_clifford_group",
]


def cy_gate(*, dtype="csr"):
    """Controlled Y gate.

    Parameters
    ----------
    dtype : str or type, [keyword only] [optional]
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    result : :class:`.Qobj`
        Quantum object for operator describing the rotation.
    """
    return Qobj(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]],
        dims=[[2, 2], [2, 2]],
    ).to(dtype)


def cz_gate(*, dtype="csr"):
    """Controlled Z gate.

    Parameters
    ----------
    dtype : str or type, [keyword only] [optional]
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    result : :class:`.Qobj`
        Quantum object for operator describing the rotation.
    """
    return qdiags([1, 1, 1, -1], dims=[[2, 2], [2, 2]], dtype=dtype)


def s_gate(*, dtype="csr"):
    """Single-qubit rotation also called Phase gate or the Z90 gate.

    Parameters
    ----------
    dtype : str or type, [keyword only] [optional]
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    result : :class:`.Qobj`
        Quantum object for operator describing
        a 90 degree rotation around the z-axis.

    """
    return qdiags([1, 1j], dtype=dtype)


def cs_gate(*, dtype="csr"):
    """Controlled S gate.

    Parameters
    ----------
    dtype : str or type, [keyword only] [optional]
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    result : :class:`.Qobj`
        Quantum object for operator describing the rotation.

    """
    return qdiags([1, 1, 1, 1j], dims=[[2, 2], [2, 2]], dtype=dtype)


def t_gate(*, dtype="csr"):
    """Single-qubit rotation related to the S gate by the relationship S=T*T.

    Parameters
    ----------
    dtype : str or type, [keyword only] [optional]
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    result : :class:`.Qobj`
        Quantum object for operator describing a phase shift of pi/4.

    """
    return qdiags([1, np.exp(1j * np.pi / 4)], dtype=dtype)


def ct_gate(*, dtype="csr"):
    """Controlled T gate.

    Parameters
    ----------
    dtype : str or type, [keyword only] [optional]
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    result : :class:`.Qobj`
        Quantum object for operator describing the rotation.

    """
    return qdiags(
        [1, 1, 1, np.exp(1j * np.pi / 4)],
        dims=[[2, 2], [2, 2]],
        dtype=dtype,
    )


def rx(phi, *, dtype="dense"):
    """Single-qubit rotation for operator sigmax with angle phi.

    Parameters
    ----------
    dtype : str or type, [keyword only] [optional]
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    result : qobj
        Quantum object for operator describing the rotation.

    """
    return Qobj(
        [
            [np.cos(phi / 2), -1j * np.sin(phi / 2)],
            [-1j * np.sin(phi / 2), np.cos(phi / 2)],
        ]
    ).to(dtype)


def ry(phi, *, dtype="dense"):
    """Single-qubit rotation for operator sigmay with angle phi.

    Parameters
    ----------
    dtype : str or type, [keyword only] [optional]
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    result : qobj
        Quantum object for operator describing the rotation.

    """
    return Qobj(
        [
            [np.cos(phi / 2), -np.sin(phi / 2)],
            [np.sin(phi / 2), np.cos(phi / 2)],
        ]
    ).to(dtype)


def rz(phi, *, dtype="csr"):
    """Single-qubit rotation for operator sigmaz with angle phi.

    Parameters
    ----------
    dtype : str or type, [keyword only] [optional]
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    result : qobj
        Quantum object for operator describing the rotation.

    """
    return qdiags([np.exp(-1j * phi / 2), np.exp(1j * phi / 2)], dtype=dtype)


def sqrtnot(*, dtype="dense"):
    """Single-qubit square root NOT gate.

    Parameters
    ----------
    dtype : str or type, [keyword only] [optional]
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    result : qobj
        Quantum object for operator describing the square root NOT gate.

    """
    return Qobj([[0.5 + 0.5j, 0.5 - 0.5j], [0.5 - 0.5j, 0.5 + 0.5j]]).to(dtype)


def snot(*, dtype="dense"):
    """Quantum object representing the SNOT (Hadamard) gate.

    Parameters
    ----------
    dtype : str or type, [keyword only] [optional]
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    snot_gate : qobj
        Quantum object representation of SNOT gate.

    """
    return Qobj([[1, 1], [1, -1]]).to(dtype) / np.sqrt(2.0)


def phasegate(theta, *, dtype="csr"):
    """
    Returns quantum object representing the phase shift gate.

    Parameters
    ----------
    theta : float
        Phase rotation angle.
    dtype : str or type, [keyword only] [optional]
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    phase_gate : qobj
        Quantum object representation of phase shift gate.

    """
    return qdiags([1, np.exp(1.0j * theta)], dtype=dtype)


def qrot(theta, phi, *, dtype="dense"):
    """
    Single qubit rotation driving by Rabi oscillation with 0 detune.

    Parameters
    ----------
    phi : float
        The inital phase of the rabi pulse.
    theta : float
        The duration of the rabi pulse.
    dtype : str or type, [keyword only] [optional]
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    qrot_gate : :class:`.Qobj`
        Quantum object representation of physical qubit rotation under
        a rabi pulse.
    """
    return Qobj(
        [
            [np.cos(theta / 2), -1j * np.exp(-1j * phi) * np.sin(theta / 2)],
            [-1j * np.exp(1j * phi) * np.sin(theta / 2), np.cos(theta / 2)],
        ]
    ).to(dtype)


#
# 2 Qubit Gates
#


def cphase(theta, *, dtype="csr"):
    """
    Returns quantum object representing the controlled phase shift gate.

    Parameters
    ----------
    theta : float
        Phase rotation angle.
    dtype : str or type, [keyword only] [optional]
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    U : qobj
        Quantum object representation of controlled phase gate.
    """
    return qdiags(
        [1, 1, 1, np.exp(1.0j * theta)], dims=[[2, 2], [2, 2]], dtype=dtype
    )


def cnot(*, dtype="csr"):
    """
    Quantum object representing the CNOT gate.

    Parameters
    ----------
    dtype : str or type, [keyword only] [optional]
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    cnot_gate : qobj
        Quantum object representation of CNOT gate

    """
    return Qobj(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dims=[[2, 2], [2, 2]],
    ).to(dtype)


def csign(*, dtype="csr"):
    """
    Quantum object representing the CSIGN gate.

    Parameters
    ----------
    dtype : str or type, [keyword only] [optional]
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    csign_gate : qobj
        Quantum object representation of CSIGN gate

    """
    return cz_gate(dtype=dtype)


def berkeley(*, dtype="dense"):
    """
    Quantum object representing the Berkeley gate.

    Parameters
    ----------
    dtype : str or type, [keyword only] [optional]
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    berkeley_gate : qobj
        Quantum object representation of Berkeley gate

    """
    return Qobj(
        [
            [np.cos(np.pi / 8), 0, 0, 1.0j * np.sin(np.pi / 8)],
            [0, np.cos(3 * np.pi / 8), 1.0j * np.sin(3 * np.pi / 8), 0],
            [0, 1.0j * np.sin(3 * np.pi / 8), np.cos(3 * np.pi / 8), 0],
            [1.0j * np.sin(np.pi / 8), 0, 0, np.cos(np.pi / 8)],
        ],
        dims=[[2, 2], [2, 2]],
    ).to(dtype)


def swapalpha(alpha, *, dtype="csr"):
    """
    Quantum object representing the SWAPalpha gate.

    Parameters
    ----------
    dtype : str or type, [keyword only] [optional]
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    swapalpha_gate : qobj
        Quantum object representation of SWAPalpha gate
    """
    phase = np.exp(1.0j * np.pi * alpha)
    return Qobj(
        [
            [1, 0, 0, 0],
            [0, 0.5 * (1 + phase), 0.5 * (1 - phase), 0],
            [0, 0.5 * (1 - phase), 0.5 * (1 + phase), 0],
            [0, 0, 0, 1],
        ],
        dims=[[2, 2], [2, 2]],
    ).to(dtype)


def swap(*, dtype="csr"):
    """Quantum object representing the SWAP gate.

    Parameters
    ----------
    dtype : str or type, [keyword only] [optional]
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    swap_gate : qobj
        Quantum object representation of SWAP gate

    """
    return Qobj(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dims=[[2, 2], [2, 2]],
    ).to(dtype)


def iswap(*, dtype="csr"):
    """Quantum object representing the iSWAP gate.

    Parameters
    ----------
    dtype : str or type, [keyword only] [optional]
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    iswap_gate : qobj
        Quantum object representation of iSWAP gate
    """
    return Qobj(
        [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]],
        dims=[[2, 2], [2, 2]],
    ).to(dtype)


def sqrtswap(*, dtype="dense"):
    """Quantum object representing the square root SWAP gate.

    Parameters
    ----------
    dtype : str or type, [keyword only] [optional]
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    sqrtswap_gate : qobj
        Quantum object representation of square root SWAP gate

    """
    return Qobj(
        np.array(
            [
                [1, 0, 0, 0],
                [0, 0.5 + 0.5j, 0.5 - 0.5j, 0],
                [0, 0.5 - 0.5j, 0.5 + 0.5j, 0],
                [0, 0, 0, 1],
            ]
        ),
        dims=[[2, 2], [2, 2]],
    ).to(dtype)


def sqrtiswap(*, dtype="dense"):
    """Quantum object representing the square root iSWAP gate.

    Parameters
    ----------
    dtype : str or type, [keyword only] [optional]
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    sqrtiswap_gate : qobj
        Quantum object representation of square root iSWAP gate
    """
    return Qobj(
        np.array(
            [
                [1, 0, 0, 0],
                [0, 1 / np.sqrt(2), 1j / np.sqrt(2), 0],
                [0, 1j / np.sqrt(2), 1 / np.sqrt(2), 0],
                [0, 0, 0, 1],
            ]
        ),
        dims=[[2, 2], [2, 2]],
    ).to(dtype)


def molmer_sorensen(theta, *, dtype="dense"):
    """
    Quantum object of a Mølmer–Sørensen gate.

    Parameters
    ----------
    theta: float
        The duration of the interaction pulse.
    N: int
        Number of qubits in the system.
    target: int
        The indices of the target qubits.
    dtype : str or type, [keyword only] [optional]
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    molmer_sorensen_gate: :class:`.Qobj`
        Quantum object representation of the Mølmer–Sørensen gate.
    """
    return Qobj(
        [
            [np.cos(theta / 2.0), 0, 0, -1.0j * np.sin(theta / 2.0)],
            [0, np.cos(theta / 2.0), -1.0j * np.sin(theta / 2.0), 0],
            [0, -1.0j * np.sin(theta / 2.0), np.cos(theta / 2.0), 0],
            [-1.0j * np.sin(theta / 2.0), 0, 0, np.cos(theta / 2.0)],
        ],
        dims=[[2, 2], [2, 2]],
    ).to(dtype)


#
# 3 Qubit Gates
#


def fredkin(*, dtype="csr"):
    """Quantum object representing the Fredkin gate.

    Parameters
    ----------
    dtype : str or type, [keyword only] [optional]
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    fredkin_gate : qobj
        Quantum object representation of Fredkin gate.

    """
    return Qobj(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ],
        dims=[[2, 2, 2], [2, 2, 2]],
    ).to(dtype)


def toffoli(*, dtype="csr"):
    """Quantum object representing the Toffoli gate.

    Parameters
    ----------
    dtype : str or type, [keyword only] [optional]
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    toff_gate : qobj
        Quantum object representation of Toffoli gate.

    """
    return Qobj(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
        ],
        dims=[[2, 2, 2], [2, 2, 2]],
    ).to(dtype)


#
# Miscellaneous Gates
#


def globalphase(theta, N=1, *, dtype="csr"):
    """
    Returns quantum object representing the global phase shift gate.

    Parameters
    ----------
    theta : float
        Phase rotation angle.

    dtype : str or type, [keyword only] [optional]
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    phase_gate : qobj
        Quantum object representation of global phase shift gate.

    """
    return qeye([2] * N, dtype=dtype) * np.exp(1.0j * theta)


#
# Operation on Gates
#


def _hamming_distance(x):
    """
    Calculate the bit-wise Hamming distance of x from 0: That is, the number
    1s in the integer x.
    """
    tot = 0
    while x:
        tot += 1
        x &= x - 1
    return tot


def hadamard_transform(N=1, *, dtype="dense"):
    """Quantum object representing the N-qubit Hadamard gate.

    Parameters
    ----------
    dtype : str or type, [keyword only] [optional]
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Returns
    -------
    q : qobj
        Quantum object representation of the N-qubit Hadamard gate.

    """
    data = 2 ** (-N / 2) * np.array(
        [
            [(-1) ** _hamming_distance(i & j) for i in range(2**N)]
            for j in range(2**N)
        ]
    )

    return Qobj(data, dims=[[2] * N, [2] * N]).to(dtype)


def _powers(op, N):
    """
    Generator that yields powers of an operator `op`,
    through to `N`.
    """
    acc = qeye_like(op)
    yield acc

    for _ in range(N - 1):
        acc *= op
        yield acc


def qubit_clifford_group(*, dtype="dense"):
    """
    Generates the Clifford group on a single qubit,
    using the presentation of the group given by Ross and Selinger
    (http://www.mathstat.dal.ca/~selinger/newsynth/).

    Parameters
    -----------
    dtype : str or type, [keyword only] [optional]
        Storage representation. Any data-layer known to `qutip.data.to` is
        accepted.

    Return
    ------
    op : list of Qobj
        Clifford operators, represented as Qobj instances.

    """

    # The Ross-Selinger presentation of the single-qubit Clifford
    # group expresses each element in the form C_{ijk} = E^i X^j S^k
    # for gates E, X and S, and for i in range(3), j in range(2) and
    # k in range(4).
    #
    # We start by defining these gates. E is defined in terms of H,
    # \omega and S, so we define \omega and H first.
    w = np.exp(1j * 2 * np.pi / 8)
    H = snot()

    X = sigmax()
    S = phasegate(np.pi / 2)
    E = H * (S**3) * w**3

    # partial(reduce, mul) returns a function that takes products
    # of its argument, by analogy to sum. Note that by analogy,
    # sum can be written as partial(reduce, add).

    # product(...) yields the Cartesian product of its arguments.
    # Here, each element is a tuple (E**i, X**j, S**k) such that
    # partial(reduce, mul) acting on the tuple yields E**i * X**j * S**k.
    return [
        op.to(dtype)
        for op in map(
            partial(reduce, mul),
            product(_powers(E, 3), _powers(X, 2), _powers(S, 4)),
        )
    ]
