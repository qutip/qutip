# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

__all__ = ['eseries', 'esval', 'esspec', 'estidy']

import numpy as np
import scipy.sparse as sp
from qutip.qobj import Qobj


class eseries():
    """
    Class representation of an exponential-series expansion of
    time-dependent quantum objects.

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

    def __init__(self, q=np.array([], dtype=object), s=np.array([])):

        if isinstance(s, (int, float, complex)):
            s = np.array([s])

        if (not np.any(np.asarray(q, dtype=object))) and (len(s) == 0):
            self.ampl = np.array([])
            self.rates = np.array([])
            self.dims = [[1, 1]]
            self.shape = [1, 1]

        elif np.any(np.asarray(q, dtype=object)) and (len(s) == 0):
            if isinstance(q, eseries):
                self.ampl = q.ampl
                self.rates = q.rates
                self.dims = q.dims
                self.shape = q.shape
            elif isinstance(q, (np.ndarray, list)):
                ind = np.shape(q)
                num = ind[0]  # number of elements in q
                sh = np.array([Qobj(x).shape for x in q])
                if any(sh != sh[0]):
                    raise TypeError('All amplitudes must have same dimension.')
                self.ampl = np.array([x for x in q])
                self.rates = np.zeros(ind)
                self.dims = self.ampl[0].dims
                self.shape = self.ampl[0].shape
            elif isinstance(q, Qobj):
                qo = Qobj(q)
                self.ampl = np.array([qo])
                self.rates = np.array([0])
                self.dims = qo.dims
                self.shape = qo.shape
            else:
                self.ampl = np.array([q])
                self.rates = np.array([0])
                self.dims = [[1, 1]]
                self.shape = [1, 1]

        elif np.any(np.asarray(q, dtype=object)) and len(s) != 0:
            if isinstance(q, (np.ndarray, list)):
                q = np.asarray(q, dtype=object)
                ind = np.shape(q)
                num = ind[0]
                sh = np.array([Qobj(q[x]).shape for x in range(0, num)])
                if np.any(sh != sh[0]):
                    raise TypeError('All amplitudes must have same dimension.')
                self.ampl = np.array([Qobj(q[x]) for x in range(0, num)],
                                     dtype=object)
                self.dims = self.ampl[0].dims
                self.shape = self.ampl[0].shape
            else:
                num = 1
                self.ampl = np.array([Qobj(q)], dtype=object)
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
            # combine arrays so that they can be sorted together
            zipped = list(zip(self.rates, self.ampl))
            zipped.sort()  # sort rates from lowest to highest
            rates, ampl = list(zip(*zipped))  # get back rates and ampl
            self.ampl = np.array(ampl, dtype=object)
            self.rates = np.array(rates)

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
            val_list = []

            for j in range(len(tlist)):
                exp_factors = np.exp(np.array(self.rates) * tlist[j])

                val = 0
                for i in range(len(self.ampl)):
                    val += self.ampl[i] * exp_factors[i]

                val_list.append(val)

            val_list = np.array(val_list, dtype=object)
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
        self.rates = np.array([])
        self.ampl = np.array([])
        for ur_key in unique_rates.keys():
            total_ampl = np.sum(np.asarray(ampl_dict[ur_key], dtype=object))

            if (isinstance(total_ampl, float) or
                    isinstance(total_ampl, complex)):
                if abs(total_ampl) > ampl_tol:
                    self.rates = np.append(self.rates, unique_rates[ur_key])
                    self.ampl = np.append(self.ampl, total_ampl)
            else:
                if abs(total_ampl.full()).max() > ampl_tol:
                    self.rates = np.append(self.rates, unique_rates[ur_key])
                    self.ampl = np.append(self.ampl,
                                          np.asarray(total_ampl,
                                                     dtype=object))

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
