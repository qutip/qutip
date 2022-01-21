import os
import numpy as np
from qutip.settings import settings as qset

from . import Qobj, QobjEvo, liouvillian, coefficient, sprepost
from ._brtools import SpectraCoefficient, _EigenBasisTransform
from ._brtensor import _BlochRedfieldElement


__all__ = ['bloch_redfield_tensor', 'brterm']


def bloch_redfield_tensor(H, a_ops, c_ops=[], sec_cutoff=0.1,
                          fock_basis=False, sparse_eigensolver=False,
                          br_dtype='sparse'):
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

        spectra : :class:`Coefficient`
            The corresponding bath spectra.
            Can be a `Coefficient` using an 'w' args or a function of the
            frequency. The `SpectraCoefficient` can be used to use array based
            coefficient. They can also depend on ``t`` if the corresponding
            ``a_op`` is a :cls:`QobjEvo`.

        Example:
            a_ops = [
                (a+a.dag(), coefficient('w>0', args={"w": 0})),
                (a+a.dag(), coefficient(lambda _, w: w>0, args={"w": 0}),
                (QobjEvo([b+b.dag(), lambda t: ...]),
                 coefficient(lambda t, w: ...), args={"w": 0}),
                (c+c.dag(), SpectraCoefficient(coefficient(array, tlist=...))),
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
        Whether to use the sparse eigensolver

    br_dtype : ['sparse', 'dense', 'data']
        Which data type to use when computing the brtensor.
        With a cutoff 'sparse' is usually the most efficient.

    Returns
    -------
    R, [evecs]: :class:`qutip.Qobj`, tuple of :class:`qutip.Qobj`
        If ``fock_basis``, return the Bloch Redfield tensor in the laboratory
        basis. Otherwise return the Bloch Redfield tensor in the diagonalized
        Hamiltonian basis and the eigenvectors of the Hamiltonian as hstacked
        column.
    """
    R = liouvillian(H, c_ops)
    H_transform = _EigenBasisTransform(QobjEvo(H), sparse_eigensolver)

    if fock_basis:
        for a_op in a_ops:
            R += brterm(H_transform, *a_op, sec_cutoff, True,
                        br_dtype=br_dtype)
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
        for a_op in a_ops:
            R += brterm(H_transform, *a_op, sec_cutoff,
                        False, br_dtype=br_dtype)[0]
        return R, H_transform.as_Qobj()


def brterm(H, a_op, spectra, sec_cutoff=0.1,
           fock_basis=False, sparse_eigensolver=False, br_dtype='sparse'):
    """
    Calculates the contribution of one coupling operator to the Bloch-Redfield
    tensor.

    Parameters
    ----------

    H : :class:`qutip.Qobj`, :class:`qutip.QobjEvo`
        System Hamiltonian.

    a_op : :class:`qutip.Qobj`, :class:`qutip.QobjEvo`
        The operator coupling to the environment. Must be hermitian.

    spectra : :class:`Coefficient`
        The corresponding bath spectra.
        Must be a :cls:`Coefficient` using an 'w' args. The
        :cls:`SpectraCoefficient` can be used to use array based coefficient.
        It can also depend on ``t`` if ``a_op`` is a :cls:`QobjEvo`.

        Example:

            coefficient('w>0', args={"w": 0})
            SpectraCoefficient(coefficient(array, tlist=...))

    sec_cutoff : float {0.1}
        Cutoff for secular approximation. Use ``-1`` if secular approximation
        is not used when evaluating bath-coupling terms.

    fock_basis : bool {False}
        Whether to return the tensor in the input basis or the diagonalized
        basis.

    sparse_eigensolver : bool {False}
        Whether to use the sparse eigensolver on the Hamiltonian.

    br_dtype : ['sparse', 'dense', 'data']
        Which data type to use when computing the brtensor.
        With a cutoff 'sparse' is usually the most efficient.

    Returns
    -------
    R, [evecs]: :class:`~Qobj`, :class:`~QobjEvo` or tuple
        If ``fock_basis``, return the Bloch Redfield tensor in the outside
        basis. Otherwise return the Bloch Redfield tensor in the diagonalized
        Hamiltonian basis and the eigenvectors of the Hamiltonian as hstacked
        column. The tensors and, if given, evecs, will be :obj:`~QobjEvo` if
        the ``H`` and ``a_op`` is time dependent, :obj:`Qobj` otherwise.
    """
    if isinstance(H, _EigenBasisTransform):
        Hdiag = H
    else:
        Hdiag = _EigenBasisTransform(QobjEvo(H), sparse=sparse_eigensolver)

    sec_cutoff = sec_cutoff if sec_cutoff >= 0 else np.inf
    R = QobjEvo(_BlochRedfieldElement(Hdiag, QobjEvo(a_op), spectra,
                sec_cutoff, not fock_basis, dtype=br_dtype))

    if (
        ((isinstance(H, _EigenBasisTransform) and H.isconstant)
         or isinstance(H, Qobj))
        and isinstance(a_op, Qobj)
    ):
        R = R(0)
    return R if fock_basis else (R, Hdiag.as_Qobj())
