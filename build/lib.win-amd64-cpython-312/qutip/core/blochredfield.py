# Required for Sphinx to follow autodoc_type_aliases
from __future__ import annotations

import os
import inspect
import numpy as np
from typing import overload, Union

from qutip.settings import settings as qset
from . import Qobj, QobjEvo, liouvillian, coefficient, sprepost
from ._brtools import SpectraCoefficient, _EigenBasisTransform
from .cy.coefficient import InterCoefficient, Coefficient
from ._brtensor import _BlochRedfieldElement, _BlochRedfieldCrossElement
from ..typing import CoeffProtocol
from .environment import BosonicEnvironment, FermionicEnvironment


__all__ = ['bloch_redfield_tensor', 'brterm', 'brcrossterm']

SpectrumType = Union[
    Coefficient, str, CoeffProtocol,
    BosonicEnvironment, FermionicEnvironment
]


@overload
def bloch_redfield_tensor(
    H: Qobj,
    a_ops: list[tuple[Qobj, SpectrumType]],
    c_ops: list[Qobj] = None,
    sec_cutoff: float = 0.1,
    fock_basis: bool = False,
    sparse_eigensolver: bool = False,
    br_computation_method: str = 'sparse',
) -> Qobj: ...

@overload
def bloch_redfield_tensor(
    H: Qobj | QobjEvo,
    a_ops: list[tuple[Qobj | QobjEvo, SpectrumType]],
    c_ops: list[Qobj | QobjEvo] = None,
    sec_cutoff: float = 0.1,
    fock_basis: bool = False,
    sparse_eigensolver: bool = False,
    br_computation_method: str = 'sparse',
) -> QobjEvo: ...

def bloch_redfield_tensor(
    H: Qobj | QobjEvo,
    a_ops: list[tuple[Qobj | QobjEvo, SpectrumType]],
    c_ops: list[Qobj | QobjEvo] = None,
    sec_cutoff: float = 0.1,
    fock_basis: bool = False,
    sparse_eigensolver: bool = False,
    br_computation_method: str = 'sparse',
) -> Qobj | QobjEvo:
    """
    Calculates the Bloch-Redfield tensor for a system given
    a set of operators and corresponding spectral functions that describes the
    system's coupling to its environment.

    Parameters
    ----------

    H : :class:`qutip.Qobj`, :class:`qutip.QobjEvo`
        System Hamiltonian.

    a_ops : list of (a_op, spectra)
        Nested list of system operators that couple to the environment,
        and the corresponding bath spectra.

        a_op : :class:`qutip.Qobj`, :class:`qutip.QobjEvo`
            The operator coupling to the environment. Must be hermitian.

        spectra : :obj:`.Coefficient`, func, str, Environment
            The corresponding bath spectra.
            Bath can be provided as :class:`.BosonicEnvironment`,
            :class:`.FermionicEnvironment` or power spectra function. These can
            be a :obj:`.Coefficient`, function or string. For coefficient, the
            frequency is passed as the 'w' args. The
            :class:`SpectraCoefficient` can be used for array based
            coefficient.

            The spectra can depend on ``t`` if ``a_op`` is a :obj:`.QobjEvo`.

        Example:

        .. code-block::

            a_ops = [
                (a+a.dag(), coefficient('w>0', args={"w": 0})),
                (QobjEvo(a+a.dag()), 'w > exp(-t)'),
                (QobjEvo([b+b.dag(), lambda t: ...]), lambda w: ...)),
                (c+c.dag(), SpectraCoefficient(coefficient(ws, tlist=ts))),
            ]


    c_ops : list
        List of system collapse operators.

    sec_cutoff : float {0.1}
        Cutoff for secular approximation. Use ``-1`` if secular approximation
        is not used when evaluating bath-coupling terms.

    fock_basis : bool {False}
        Whether to return the tensor in the input basis or the diagonalized
        basis.

    sparse_eigensolver : bool {False}
        Deprecated

    br_computation_method : ['sparse', 'dense', 'matrix']
        How computation for the tensor is made.
        With 'dense' or 'sparse', the secular cutoff is checked first and only
        elements within the approximation are computed and stored in an
        (sparse) array.
        With 'matrix', the tensor is build using matrix operation and the
        secular cutoff is applied at the end.

    Returns
    -------
    R, [evecs]: :class:`qutip.Qobj`, tuple of :class:`qutip.Qobj`
        If ``fock_basis``, return the Bloch Redfield tensor in the laboratory
        basis. Otherwise return the Bloch Redfield tensor in the diagonalized
        Hamiltonian basis and the eigenvectors of the Hamiltonian as hstacked
        column.
    """
    R = liouvillian(H, c_ops)
    H_transform = _EigenBasisTransform(QobjEvo(H), sparse=sparse_eigensolver)

    if fock_basis:
        for (a_op, spectra) in a_ops:
            if isinstance(spectra, FermionicEnvironment):
                R += brcrossterm(
                    H_transform, a_op, a_op.dag(),
                    lambda w: spectra.power_spectrum_plus(-w), sec_cutoff,
                    True, br_computation_method=br_computation_method
                )
                R += brcrossterm(
                    H_transform, a_op.dag(), a_op,
                    spectra.power_spectrum_minus, sec_cutoff, True,
                    br_computation_method=br_computation_method
                )
            else:
                R += brterm(H_transform, a_op, spectra, sec_cutoff, True,
                            br_computation_method=br_computation_method)
        return R
    else:
        # When the Hamiltonian is time-dependent, the transformation of `L` to
        # eigenbasis is not optimized.
        if isinstance(R, QobjEvo):
            # The `sprepost` will be computed 2 times for each parts of `R`.
            # Compressing the QobjEvo will lower the number of parts.
            R.compress()
        evec = H_transform.as_Qobj()
        R = sprepost(evec, evec.dag()) @ R @ sprepost(evec.dag(), evec)
        for (a_op, spectra) in a_ops:
            if isinstance(spectra, FermionicEnvironment):
                R += brcrossterm(
                    H_transform, a_op, a_op.dag(),
                    lambda w: spectra.power_spectrum_plus(-w), sec_cutoff,
                    False, br_computation_method=br_computation_method
                )[0]
                R += brcrossterm(
                    H_transform, a_op.dag(), a_op,
                    spectra.power_spectrum_minus, sec_cutoff, False,
                    br_computation_method=br_computation_method
                )[0]
            else:
                R += brterm(H_transform, a_op, spectra, sec_cutoff, False,
                            br_computation_method=br_computation_method)[0]
        return R, H_transform.as_Qobj()

@overload
def brterm(
    H: Qobj,
    a_op: Qobj,
    spectra: SpectrumType,
    sec_cutoff: float = 0.1,
    fock_basis: bool = False,
    sparse_eigensolver: bool = False,
    br_computation_method: str = 'sparse',
) -> Qobj: ...

@overload
def brterm(
    H: Qobj | QobjEvo,
    a_op: Qobj | QobjEvo,
    spectra: SpectrumType,
    sec_cutoff: float = 0.1,
    fock_basis: bool = False,
    sparse_eigensolver: bool = False,
    br_computation_method: str = 'sparse',
) -> QobjEvo: ...

def brterm(
    H: Qobj | QobjEvo,
    a_op: Qobj | QobjEvo,
    spectra: SpectrumType,
    sec_cutoff: float = 0.1,
    fock_basis: bool = False,
    sparse_eigensolver: bool = False,
    br_computation_method: str = 'sparse',
) -> Qobj | QobjEvo:
    """
    Calculates the contribution of one coupling operator to the Bloch-Redfield
    tensor.

    Parameters
    ----------

    H : :class:`qutip.Qobj`, :class:`qutip.QobjEvo`
        System Hamiltonian.

    a_op : :class:`qutip.Qobj`, :class:`qutip.QobjEvo`
        The operator coupling to the environment. Expected to be hermitian.

    spectra : :obj:`.Coefficient`, func, str, Environment
        The corresponding bath spectra.
        Bath can be provided as :class:`.BosonicEnvironment`,
        :class:`.FermionicEnvironment` or power spectra function. These can be
        a :obj:`.Coefficient`, function or string. For coefficient, the
        frequency is passed as the 'w' args. The :class:`SpectraCoefficient`
        can be used for array based coefficient.

        The spectra can depend on ``t`` if ``a_op`` is a :obj:`.QobjEvo`.

        Example:

            coefficient('w>0', args={"w": 0})
            SpectraCoefficient(coefficient(array, tlist=...))
            lambda w: (w > 0) * 1.
            BosonicEnvironment(...)

    sec_cutoff : float {0.1}
        Cutoff for secular approximation. Use ``-1`` if secular approximation
        is not used when evaluating bath-coupling terms.

    fock_basis : bool {False}
        Whether to return the tensor in the input basis or the diagonalized
        basis.

    sparse_eigensolver : bool {False}
        Deprecated

    br_computation_method : ['sparse', 'dense', 'matrix']
        How computation for the tensor is made.
        With 'dense' or 'sparse', the secular cutoff is checked first and only
        elements within the approximation are computed and stored in an
        (sparse) array.
        With 'matrix', the tensor is build using matrix operation and the
        secular cutoff is applied at the end.

    Returns
    -------
    R, [evecs]: :obj:`.Qobj`, :obj:`.QobjEvo` or tuple
        If ``fock_basis``, return the Bloch Redfield tensor in the outside
        basis. Otherwise return the Bloch Redfield tensor in the diagonalized
        Hamiltonian basis and the eigenvectors of the Hamiltonian as hstacked
        column. The tensors and, if given, evecs, will be :obj:`.QobjEvo` if
        the ``H`` and ``a_op`` is time dependent, :obj:`.Qobj` otherwise.
    """
    if isinstance(H, _EigenBasisTransform):
        Hdiag = H
    else:
        Hdiag = _EigenBasisTransform(QobjEvo(H), sparse=sparse_eigensolver)

    # convert spectra to Coefficient
    if isinstance(spectra, str):
        spectra = coefficient(spectra, args={'w': 0})
    elif isinstance(spectra, InterCoefficient):
        spectra = SpectraCoefficient(spectra)
    elif isinstance(spectra, BosonicEnvironment):
        spectra = SpectraCoefficient(coefficient(spectra.power_spectrum))
    elif isinstance(spectra, Coefficient):
        pass
    elif callable(spectra):
        sig = inspect.signature(spectra)
        if tuple(sig.parameters.keys()) == ("w",):
            spectra = SpectraCoefficient(coefficient(spectra))
        else:
            spectra = coefficient(spectra, args={'w': 0})
    else:
        raise TypeError("a_ops's spectra not known")

    sec_cutoff = sec_cutoff if sec_cutoff >= 0 else np.inf
    R = QobjEvo(_BlochRedfieldElement(Hdiag, QobjEvo(a_op), spectra,
                sec_cutoff, not fock_basis, dtype=br_computation_method))

    if (
        ((isinstance(H, _EigenBasisTransform) and H.isconstant)
         or isinstance(H, Qobj))
        and isinstance(a_op, Qobj)
    ):
        R = R(0)
    return R if fock_basis else (R, Hdiag.as_Qobj())


@overload
def brcrossterm(
    H: Qobj,
    a_op: Qobj,
    b_op: Qobj,
    spectra: Coefficient | CoeffProtocol | str,
    sec_cutoff: float = 0.1,
    fock_basis: bool = False,
    sparse_eigensolver: bool = False,
    br_computation_method: str = 'sparse',
) -> Qobj: ...

@overload
def brcrossterm(
    H: Qobj | QobjEvo,
    a_op: Qobj | QobjEvo,
    b_op: Qobj | QobjEvo,
    spectra: Coefficient | CoeffProtocol | str,
    sec_cutoff: float = 0.1,
    fock_basis: bool = False,
    sparse_eigensolver: bool = False,
    br_computation_method: str = 'sparse',
) -> QobjEvo: ...

def brcrossterm(
    H: Qobj | QobjEvo,
    a_op: Qobj | QobjEvo,
    b_op: Qobj | QobjEvo,
    spectra: Coefficient | CoeffProtocol | str,
    sec_cutoff: float = 0.1,
    fock_basis: bool = False,
    sparse_eigensolver: bool = False,
    br_computation_method: str = 'sparse',
) -> Qobj | QobjEvo:
    """
    Calculates the contribution of one coupling operator to the Bloch-Redfield
    tensor.

    Parameters
    ----------

    H : :class:`qutip.Qobj`, :class:`qutip.QobjEvo`
        System Hamiltonian.

    a_op : :class:`qutip.Qobj`, :class:`qutip.QobjEvo`
        The operator coupling to the environment.

    b_op : :class:`qutip.Qobj`, :class:`qutip.QobjEvo`
        The operator coupling to the environment.

    spectra : :obj:`.Coefficient`, func, str, Environment
        The corresponding bath spectra.
        Bath can be provided as :class:`.BosonicEnvironment`,
        :class:`.FermionicEnvironment` or power spectra function. These can be
        a :obj:`.Coefficient`, function or string. For coefficient, the
        frequency is passed as the 'w' args. The :class:`SpectraCoefficient`
        can be used for array based coefficient.

        The spectra can depend on ``t`` if ``a_op`` is a :obj:`.QobjEvo`.

        Example:

            coefficient('w>0', args={"w": 0})
            SpectraCoefficient(coefficient(array, tlist=...))
            lambda w: (w > 0) * 1.
            BosonicEnvironment(...)

    sec_cutoff : float {0.1}
        Cutoff for secular approximation. Use ``-1`` if secular approximation
        is not used when evaluating bath-coupling terms.

    fock_basis : bool {False}
        Whether to return the tensor in the input basis or the diagonalized
        basis.

    sparse_eigensolver : bool {False}
        Deprecated

    br_computation_method : ['dense', 'sparse', 'matrix']
        How computation for the tensor is made.
        With 'dense' or 'sparse', the secular cutoff is checked first and only
        elements within the approximation are computed and stored in an
        (sparse) array.
        With 'matrix', the tensor is build using matrix operation and the
        secular cutoff is applied at the end.

    Returns
    -------
    R, [evecs]: :obj:`.Qobj`, :obj:`.QobjEvo` or tuple
        If ``fock_basis``, return the Bloch Redfield tensor in the outside
        basis. Otherwise return the Bloch Redfield tensor in the diagonalized
        Hamiltonian basis and the eigenvectors of the Hamiltonian as hstacked
        column. The tensors and, if given, evecs, will be :obj:`.QobjEvo` if
        the ``H`` and ``a_op`` is time dependent, :obj:`.Qobj` otherwise.
    """
    if isinstance(H, _EigenBasisTransform):
        Hdiag = H
    else:
        Hdiag = _EigenBasisTransform(QobjEvo(H), sparse=sparse_eigensolver)

    # convert spectra to Coefficient
    if isinstance(spectra, str):
        spectra = coefficient(spectra, args={'w': 0})
    elif isinstance(spectra, InterCoefficient):
        spectra = SpectraCoefficient(spectra)
    elif isinstance(spectra, Coefficient):
        pass
    elif callable(spectra):
        sig = inspect.signature(spectra)
        if tuple(sig.parameters.keys()) == ("w",):
            spectra = SpectraCoefficient(coefficient(spectra))
        else:
            spectra = coefficient(spectra, args={'w': 0})
    else:
        raise TypeError("a_ops's spectra not known")

    sec_cutoff = sec_cutoff if sec_cutoff >= 0 else np.inf
    R = QobjEvo(_BlochRedfieldCrossElement(
        Hdiag, QobjEvo(a_op), QobjEvo(b_op),
        spectra, sec_cutoff, not fock_basis, dtype=br_computation_method
    ))

    if (
        ((isinstance(H, _EigenBasisTransform) and H.isconstant)
         or isinstance(H, Qobj))
        and isinstance(a_op, Qobj)
    ):
        R = R(0)
    return R if fock_basis else (R, Hdiag.as_Qobj())
