__all__ = ['eseries', 'esval', 'esspec', 'estidy']

import numpy as np
import scipy.sparse as sp
from qutip.qobj import Qobj

import warnings


class eseries():
    """
    Class representation of an exponential-series expansion of
    time-dependent quantum objects.

    .. deprecated:: 4.6.0
        :obj:`~eseries` will be removed in QuTiP 5.
        Please use :obj:`~qutip.QobjEvo` for general time-dependence.

    Attributes
    ----------
    ampl : ndarray
        Array of amplitudes for exponential series.
    rates : ndarray
        Array of rates for exponential series.
    dims : list
        Dimensions of exponential series components
    shape : list
        Shape corresponding to exponential series components

    Methods
    -------
    value(tlist)
        Evaluate an exponential series at the times listed in tlist
    spec(wlist)
        Evaluate the spectrum of an exponential series at frequencies in wlist.
    tidyup()
        Returns a tidier version of the exponential series

    """
    __array_priority__ = 101

    def __init__(self, q=None, s=np.array([])):
        warnings.warn(
            "eseries is to be removed in QuTiP 5.0,"
            " consider swapping to QobjEvo for general time dependence.",
            DeprecationWarning, stacklevel=2,
        )

        if isinstance(s, (int, float, complex)):
            s = np.array([s])

        if q is None:
            self.ampl = np.array([])
            self.rates = np.array([])
            self.dims = [[1, 1]]
            self.shape = [1, 1]

        elif (len(s) == 0):
            if isinstance(q, eseries):
                self.ampl = q.ampl
                self.rates = q.rates
                self.dims = q.dims
                self.shape = q.shape
            elif isinstance(q, (np.ndarray, list)):
                num = len(q)  # number of elements in q
                if any([Qobj(x).shape != Qobj(q[0]).shape for x in q]):
                    raise TypeError('All amplitudes must have same dimension.')
                self.ampl = np.empty((num,), dtype=object)
                self.ampl[:] = q
                self.rates = np.zeros((num,))
                self.dims = self.ampl[0].dims
                self.shape = self.ampl[0].shape
            elif isinstance(q, Qobj):
                qo = Qobj(q)
                self.ampl = np.empty((1,), dtype=object)
                self.ampl[0] = qo
                self.rates = np.array([0])
                self.dims = qo.dims
                self.shape = qo.shape
            else:
                self.ampl = np.array([q])
                self.rates = np.array([0])
                self.dims = [[1, 1]]
                self.shape = [1, 1]

        elif len(s) != 0:
            if isinstance(q, (np.ndarray, list)):
                num = len(q)
                if any([Qobj(x).shape != Qobj(q[0]).shape for x in q]):
                    raise TypeError('All amplitudes must have same dimension.')
                self.ampl = np.empty((num,), dtype=object)
                self.ampl[:] = [Qobj(qq) for qq in q]
                self.dims = self.ampl[0].dims
                self.shape = self.ampl[0].shape
            else:
                num = 1
                self.ampl = np.empty((num,), dtype=object)
                self.ampl[0] = Qobj(q)
                self.dims = self.ampl[0].dims
                self.shape = self.ampl[0].shape

            if isinstance(s, (int, complex, float)):
                if num != 1:
                    raise TypeError('Number of rates must match number ' +
                                    'of members in object array.')
                self.rates = np.array([s])
            elif isinstance(s, (np.ndarray, list)):
                if len(s) != num:
                    raise TypeError('Number of rates must match number ' +
                                    ' of members in object array.')
                self.rates = np.array(s)

        if len(self.ampl) != 0:
            # Sort rates from lowest to highest.
            order = np.argsort(self.rates)
            self.ampl, self.rates = self.ampl[order], self.rates[order]

    def __str__(self):  # string of ESERIES information
        self.tidyup()
        s = "ESERIES object: " + str(len(self.ampl)) + " terms\n"
        s += "Hilbert space dimensions: " + str(self.dims) + "\n"
        for k in range(0, len(self.ampl)):
            s += "Exponent #" + str(k) + " = " + str(self.rates[k]) + "\n"
            if isinstance(self.ampl[k], sp.spmatrix):
                s += str(self.ampl[k]) + "\n"
            else:
                s += str(self.ampl[k]) + "\n"
        return s

    def __repr__(self):
        return self.__str__()

    # Addition with ESERIES on left (ex. ESERIES+5)
    def __add__(self, other):
        right = eseries(other)
        if self.dims != right.dims:
            raise TypeError("Incompatible operands for ESERIES addition")
        out = eseries()
        out.dims = self.dims
        out.shape = self.shape
        out.ampl = np.append(self.ampl, right.ampl)
        out.rates = np.append(self.rates, right.rates)
        return out

    # Addition with ESERIES on right(ex. 5+ESERIES)
    def __radd__(self, other):
        return self + other

    # define negation of ESERIES
    def __neg__(self):
        out = eseries()
        out.dims = self.dims
        out.shape = self.shape
        out.ampl = -self.ampl
        out.rates = self.rates
        return out

    # Subtraction with ESERIES on left (ex. ESERIES-5)
    def __sub__(self, other):
        return self + (-other)

    # Subtraction with ESERIES on right (ex. 5-ESERIES)
    def __rsub__(self, other):
        return other + (-self)

    # Multiplication with ESERIES on left (ex. ESERIES*other)
    def __mul__(self, other):

        if isinstance(other, eseries):
            out = eseries()
            out.dims = self.dims
            out.shape = self.shape

            for i in range(len(self.rates)):
                for j in range(len(other.rates)):
                    out += eseries(self.ampl[i] * other.ampl[j],
                                   self.rates[i] + other.rates[j])

            return out
        else:
            out = eseries()
            out.dims = self.dims
            out.shape = self.shape
            out.ampl = self.ampl * other
            out.rates = self.rates
            return out

    # Multiplication with ESERIES on right (ex. other*ESERIES)
    def __rmul__(self, other):
        out = eseries()
        out.dims = self.dims
        out.shape = self.shape
        out.ampl = other * self.ampl
        out.rates = self.rates
        return out

    #
    # todo:
    # select_ampl, select_rate: functions to select some terms given the ampl
    # or rate. This is done with {ampl} or (rate) in qotoolbox. we should use
    # functions with descriptive names for this.
    #

    #
    # evaluate the eseries for a list of times
    #
    def value(self, tlist):
        """
        Evaluates an exponential series at the times listed in ``tlist``.

        Parameters
        ----------
        tlist : ndarray
            Times at which to evaluate exponential series.

        Returns
        -------
        val_list : ndarray
            Values of exponential at times in ``tlist``.

        """
        if self.ampl is None or len(self.ampl) == 0:
            # no terms, evalue to zero
            return np.zeros(np.shape(tlist))

        if isinstance(tlist, float) or isinstance(tlist, int):
            tlist = [tlist]

        if isinstance(self.ampl[0], Qobj):
            # amplitude vector contains quantum objects
            val_list = np.empty((len(tlist),), dtype=object)

            for j in range(len(tlist)):
                exp_factors = np.exp(np.array(self.rates) * tlist[j])
                val = 0
                for i in range(len(self.ampl)):
                    val += self.ampl[i] * exp_factors[i]
                val_list[j] = val
        else:
            # the amplitude vector contains c numbers
            val_list = np.zeros(np.size(tlist), dtype=complex)
            for j in range(len(tlist)):
                exp_factors = np.exp(np.array(self.rates) * tlist[j])
                val_list[j] = np.sum(np.dot(self.ampl, exp_factors))

        if all(np.imag(val_list) == 0):
            val_list = np.real(val_list)
        if len(tlist) == 1:
            return val_list[0]
        else:
            return val_list

    def spec(self, wlist):
        """
        Evaluate the spectrum of an exponential series at frequencies
        in ``wlist``.

        Parameters
        ----------
        wlist : array_like
            Array/list of frequenies.

        Returns
        -------
        val_list : ndarray
            Values of exponential series at frequencies in ``wlist``.

        """
        val_list = np.zeros(np.size(wlist))

        for i in range(len(wlist)):
            val_list[i] = 2 * np.real(
                np.dot(self.ampl, 1. / (1.0j * wlist[i] - self.rates)))

        return val_list

    def tidyup(self, *args):
        """ Returns a tidier version of exponential series.
        """
        #
        # combine duplicate entries (same rate)
        #
        rate_tol = 1e-10
        ampl_tol = 1e-10

        ampl_dict = {}
        unique_rates = {}
        ur_len = 0

        for r_idx in range(len(self.rates)):
            # look for a matching rate in the list of unique rates
            idx = -1
            for ur_key in unique_rates.keys():
                if abs(self.rates[r_idx] - unique_rates[ur_key]) < rate_tol:
                    idx = ur_key
                    break
            if idx == -1:
                # no matching rate, add it
                unique_rates[ur_len] = self.rates[r_idx]
                ampl_dict[ur_len] = [self.ampl[r_idx]]
                ur_len = len(unique_rates)
            else:
                # found matching rate, append amplitude to its list
                ampl_dict[idx].append(self.ampl[r_idx])

        # create new amplitude and rate list with only unique rates, and
        # nonzero amplitudes
        rates, ampl = [], []
        for ur_key in unique_rates.keys():
            total_ampl = sum(ampl_dict[ur_key])
            if (isinstance(total_ampl, float) or
                    isinstance(total_ampl, complex)):
                if abs(total_ampl) > ampl_tol:
                    rates.append(unique_rates[ur_key])
                    ampl.append(total_ampl)
            else:
                if abs(total_ampl.full()).max() > ampl_tol:
                    rates.append(unique_rates[ur_key])
                    ampl.append(total_ampl)
        self.rates = np.array(rates)
        self.ampl = np.empty((len(ampl),), dtype=object)
        self.ampl[:] = ampl
        return self


# -----------------------------------------------------------------------------
#
# wrapper functions for accessing the class methods (for compatibility with
# quantum optics toolbox)
#
def esval(es, tlist):
    """
    Evaluates an exponential series at the times listed in ``tlist``.

    Parameters
    ----------
    tlist : ndarray
        Times at which to evaluate exponential series.

    Returns
    -------
    val_list : ndarray
        Values of exponential at times in ``tlist``.

    """
    return es.value(tlist)


def esspec(es, wlist):
    """Evaluate the spectrum of an exponential series at frequencies
    in ``wlist``.

    Parameters
    ----------
    wlist : array_like
        Array/list of frequenies.

    Returns
    -------
    val_list : ndarray
        Values of exponential series at frequencies in ``wlist``.

    """
    return es.spec(wlist)


def estidy(es, *args):
    """
    Returns a tidier version of exponential series.
    """
    return es.tidyup()
