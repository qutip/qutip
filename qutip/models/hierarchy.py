"""
Generic Hierarchy equations of motion model for a sum of exponentials
"""
import numpy as np

from copy import copy
from numpy import matrix
from numpy import linalg

from scipy.misc import factorial
from scipy.optimize import curve_fit
from scipy.constants import k
import scipy.sparse as sp
import scipy.integrate
from scipy.integrate import quad
from scipy.signal import hilbert

from qutip.cy.spmatfuncs import cy_ode_rhs
from qutip import Qobj
from qutip import spre, spost, sprepost, thermal_dm, mesolve, Options, dims
from qutip import tensor, identity, destroy, sigmax, sigmaz, basis, qeye
from qutip import liouvillian as liouv
from qutip import mat2vec, state_number_enumerate, basis
from qutip.ui.progressbar import BaseProgressBar, TextProgressBar
from qutip.solver import Result

import numpy as np

def _heom_state_dictionaries(dims, excitations):
    """
    Return the number of states, and lookup-dictionaries for translating
    a state tuple to a state index, and vice versa, for a system with a given
    number of components and maximum number of excitations.
    Parameters
    ----------
    dims: list
        A list with the number of states in each sub-system.
    excitations : integer
        The maximum numbers of dimension
    Returns
    -------
    nstates, state2idx, idx2state: integer, dict, dict
        The number of states `nstates`, a dictionary for looking up state
        indices from a state tuple, and a dictionary for looking up state
        state tuples from state indices.
    """
    nstates = 0
    state2idx = {}
    idx2state = {}

    for state in state_number_enumerate(dims, excitations):
        state2idx[state] = nstates
        idx2state[nstates] = state
        nstates += 1

    return nstates, state2idx, idx2state


def _heom_number_enumerate(dims, excitations=None, state=None, idx=0):
    """
    An iterator that enumerates all the state number arrays (quantum numbers on
    the form [n1, n2, n3, ...]) for a system with dimensions given by dims.
    Example:
        >>> for state in state_number_enumerate([2,2]):
        >>>     print(state)
        [ 0.  0.]
        [ 0.  1.]
        [ 1.  0.]
        [ 1.  1.]
    Parameters
    ----------
    dims : list or array
        The quantum state dimensions array, as it would appear in a Qobj.
    state : list
        Current state in the iteration. Used internally.
    excitations : integer (None)
        Restrict state space to states with excitation numbers below or
        equal to this value.
    idx : integer
        Current index in the iteration. Used internally.
    Returns
    -------
    state_number : list
        Successive state number arrays that can be used in loops and other
        iterations, using standard state enumeration *by definition*.
    """

    if state is None:
        state = np.zeros(len(dims))

    if excitations and sum(state[0:idx]) > excitations:
        pass

    elif idx == len(dims):
        if excitations is None:
            yield np.array(state)
        else:
            yield tuple(state)

    else:
        for n in range(dims[idx]):
            state[idx] = n
            for s in state_number_enumerate(dims, excitations, state, idx + 1):
                yield s


def cot(x):
    """
    Calculate cotangent.
    Parameters
    ----------
    x: Float
        Angle.
    """
    return np.cos(x) / np.sin(x)


class Heom(object):
    """
    The hierarchy model class.

    Parameters
    ==========
    hamiltonian: Qobj
        The system hamiltonian

    coupling: float
        The coupling strength

    temperature: float
        The temperature of the bath in units corresponding to the planck const
        default: 1.

    Nc: int
        Cutoff temperature for the bath
        default=1

    real_coeff: list
        The list of coefficients for the real terms in the exponential
        series expansion of the spectral density

    real_freq: list
        The list of frequencies for the real terms of the exponential series
        expansion of the spectral density

    imaginary_coeff: list
        The list of coefficients for the imaginary terms of the exponential
        series expansion of the spectral density

    imaginary_freq: list
        The list of frequencies for the imaginary terms of the exponential
        series expansion of the spectral density

    planck: float
        default: 1.
        The reduced Planck's constant.

    boltzmann: float
        default: 1.
        The reduced Boltzmann's constant.
    """
    def __init__(self, hamiltonian, coupling, temperature = 1, Nc = 1,
                 real_coeff=[], real_freq=[],
                 imaginary_coeff=[], imaginary_freq=[],
                 planck=1., boltzmann=1.):
        self.hamiltonian = hamiltonian
        self.coupling = coupling
        self.temperature = temperature
        self.Nc = Nc

        # Make this automated from just a normal list of real+complex list
        # Have a check on the lengths of the two lists if this is separate

        self.real_prefactors = real_coeff
        self.real_exponents = real_freq
        self.complex_prefactors = imaginary_coeff
        self.complex_exponents = imaginary_freq
        
        self.planck = planck
        self.boltzmann = boltzmann
        self.NR = len(real_coeff)
        self.NI = len(imaginary_coeff)
        self._N_he = 0

        self.progress_bar = None

    def _rhs(self, progress_bar=False):
        """
        Construct the RHS for the dynamics of this system
        """
        NI = self.NI
        NR = self.NR
        N = NI + NR
        Nc = self.Nc
        Q = self.coupling
        ckAR = self.real_prefactors
        ckAI = self.complex_prefactors
        vkAR = self.real_exponents
        vkAI = self.complex_exponents

        if isinstance(progress_bar, BaseProgressBar):
            self.progress_bar = progress_bar
        elif progress_bar == True:
            self.progress_bar = TextProgressBar()
        elif progress_bar == False:
            self.progress_bar = None 
        # Set by system
        N_temp = 1
        for i in self.hamiltonian.dims[0]:
            N_temp *= N_temp*i
        sup_dim = N_temp**2
        unit = qeye(N_temp)

        # Ntot is the total number of ancillary elements in the hierarchy
        Ntot = int(round(factorial(Nc + N) / (factorial(Nc) * factorial(N))))
        # LD1 = -2.* spre(Q) * spost(Q.dag()) + spre(Q.dag() * Q) + spost(Q.dag() * Q)

        # L12=0.*LD1;
        # Setup liouvillian
        L = liouv(self.hamiltonian, [])
        Ltot = L.data
        unitthing = sp.identity(Ntot, dtype='complex', format='csr')
        Lbig = sp.kron(unitthing, Ltot.tocsr())
        
        nstates, state2idx, idx2state =_heom_state_dictionaries([Nc+1]*(N),Nc)
        self._N_he = nstates
        for nlabelt in _heom_number_enumerate([Nc+1]*(N),Nc):
            nlabel = list(nlabelt)                    
            ntotalcheck = 0
            for ncheck in range(N):
                ntotalcheck = ntotalcheck + nlabel[ncheck]                            
            current_pos = int(round(state2idx[tuple(nlabel)]))
            Ltemp = sp.lil_matrix((Ntot, Ntot))
            Ltemp[current_pos,current_pos] = 1
            Ltemp.tocsr()
            for idx,vki in enumerate(vkAR):
                Lbig = Lbig + sp.kron(Ltemp,(-nlabel[idx] * vki * spre(unit).data))

            #treat real and imaginary parts seperately

            for idx,vki in enumerate(vkAI):
                Lbig = Lbig + sp.kron(Ltemp,(-nlabel[(NR)+idx] * vki * spre(unit).data))
                #Lbig = Lbig + sp.kron(Ltemp,(-nlabel[1] * vkA[1] * spre(unit).data))
            #for kcount in range(N):
            #    Lbig = Lbig + sp.kron(Ltemp,(-nlabel[kcount] * (vkA[kcount])
            #                    * spre(unit).data))

            for kcount in range(N):
                if nlabel[kcount]>=1:
                #find the position of the neighbour
                    nlabeltemp = copy(nlabel)
                    nlabel[kcount] = nlabel[kcount] -1
                    current_pos2 = int(round(state2idx[tuple(nlabel)]))
                    Ltemp = sp.lil_matrix((Ntot, Ntot))
                    Ltemp[current_pos, current_pos2] = 1
                    Ltemp.tocsr()
                # renormalized version:    
                    #ci =  (4 * lam0 * gam * kb * Temperature * kcount
                    #      * gj/((kcount * gj)**2 - gam**2)) / (hbar**2)
                    if kcount<=(NR-1):
                        #DO REAL PARTS Vx
                        c0=ckAR[kcount]
                        c0n=ckAR[kcount]
                        Lbig = Lbig + sp.kron(Ltemp,(-1.j
                                         * np.sqrt((nlabeltemp[kcount]
                                            / abs(c0n)))
                                         * c0*(spre(Q).data - spost(Q).data)))
                    if kcount>=(NR):     
                        #in=lam
                        #i =  ckA[kcount]
                        c0=ckAI[kcount-NR]
                        c0n=ckAI[kcount-NR]
                        Lbig = Lbig + sp.kron(Ltemp,(-1.j
                                         * np.sqrt((nlabeltemp[kcount]
                                            / abs(c0n)))
                                         * (1.j*(c0) * (spre(Q).data + spost(Q).data))))
                    nlabel = copy(nlabeltemp)

            for kcount in range(N):
                if ntotalcheck<=(Nc-1):
                    nlabeltemp = copy(nlabel)
                    nlabel[kcount] = nlabel[kcount] + 1
                    current_pos3 = int(round(state2idx[tuple(nlabel)]))
                if current_pos3<=(Ntot):
                    Ltemp = sp.lil_matrix((Ntot, Ntot))
                    Ltemp[current_pos, current_pos3] = 1
                    Ltemp.tocsr()
                #renormalized   
                    if kcount<=(NR-1):
                        c0n=ckAR[kcount]

                        Lbig = Lbig + sp.kron(Ltemp,-1.j
                                      * np.sqrt((nlabeltemp[kcount]+1)*((abs(c0n))))
                                      * (spre(Q)- spost(Q)).data)
                    if kcount>=(NR):
                        cin=ckAI[kcount-NR]

                        Lbig = Lbig + sp.kron(Ltemp,-1.j
                                      * np.sqrt((nlabeltemp[kcount]+1)*(abs(cin)))
                                      * (spre(Q)- spost(Q)).data)
                nlabel = copy(nlabeltemp)

        rhs = Lbig.tocsr()
        return rhs



    def solve(self, initial_state, tlist, options=None, progress_bar=False):
        """
        """
        if options is None:
            options = Options()

        if isinstance(progress_bar, BaseProgressBar):
            self.progress_bar = progress_bar
        elif progress_bar == True:
            self.progress_bar = TextProgressBar()
        elif progress_bar == False:
            self.progress_bar = None

        N = self.NI + self.NR
        Nc = self.Nc
        rho0 = initial_state
        # Set by system
        N_temp = 1
        for i in self.hamiltonian.dims[0]:
            N_temp = N_temp*i

        sup_dim = N_temp**2
        unit = qeye(N_temp)

        L_helems = self._rhs(progress_bar)
        output = Result()
        output.solver = "hsolve"
        output.times = tlist
        output.states = []
        output.states.append(Qobj(rho0))

        rho0_flat = rho0.full().ravel('F') # Using 'F' effectively transposes
        rho0_he = np.zeros([sup_dim*self._N_he], dtype=complex)
        rho0_he[:sup_dim] = rho0_flat

        # Ntot is the total number of ancillary elements in the hierarchy
        r = scipy.integrate.ode(cy_ode_rhs)
        r.set_f_params(L_helems.data, L_helems.indices, L_helems.indptr)
        r.set_integrator('zvode', method=options.method, order=options.order,
                         atol=options.atol, rtol=options.rtol,
                         nsteps=options.nsteps, first_step=options.first_step,
                         min_step=options.min_step, max_step=options.max_step)
        r.set_initial_value(rho0_he, tlist[0])
        dt = np.diff(tlist)
        n_tsteps = len(tlist)
        for t_idx, t in enumerate(tlist):
            if t_idx < n_tsteps - 1:
                r.integrate(r.t + dt[t_idx])
                rho = Qobj(r.y[:sup_dim].reshape(rho0.shape), dims=rho0.dims)
                output.states.append(rho)
        return output


def lorrentz(w, gamma = 0.05, lam = 0.1, w0 = 1, wc = None):
    """
    The lorrentz spectral density.
    
    Parameters
    ==========
    w: array
        A numpy array with the angular frequencies for the SD
        
    gamma: float
        The value of the broadening term in the Lorrentzian
    
    lam: float
        The amplitude term of the Lorrentzian
        
    w0: float
        The single mode frequency of the cavity
        
    wc: float
        The cut off frequency to force the spectral density to fall
        to zero at large temperatures
        default: None
        
    Returns
    =======
    J: array
        The spectral density J(w) for given parameters. If the cutoff is
        specified then the spectral density is multiplied by exp(-w/wc)
    """
    num = lam*gamma
    den = ((w - w0)**2 + gamma**2)*(np.pi)
    if wc:
        return (num/den)*(np.exp(-w/wc))
    else:
        return(num/den)

def coth(x):
    """
    The coth function
    """
    return np.cosh(x)/np.sinh(x)

def exp_sum(x, amplitudes, freq):
    """
    The sum of exponentials for a given array of amplitudes and frequencies
    """
    total = 0
    for (c, nu) in zip(amplitudes, freq):
        total += c*np.exp(nu*x)
    return total

def bath_correlation(tlist, spectrum, params, wstart = 0., wend = 1.,
                     temperature = 0):
    """
    The correlation function calculated for the specific spectrum at a given
    temperature of the bath. At zero temperature the coth(x) is set to 1 for
    the real part of the correlation function integral
    
    Parameters
    ==========
    tlist: ndarray
        A 1D array with the times to calculate the correlation at

    spectrum: callable
        A function of the form f(w, *params) which calculates the spectral
        densities for the given parameters. For example, this could be set
        to `lorrentz`
        
    params: ndarray
        A 1D array of parameters for the spectral density function.
        `[gamma, lam, w0, wc]`
    
    wstart, wend: float
        The starting and ending value of the angular frequencies for 
        integration. In general the intergration is for all values but
        since at higher frequencies, the spectral density is zero, we set
        a finite limit to the numberical integration
    
    temperature: float
        The absolute temperature of the bath in Kelvin. If the temperature
        is set to zero, we can replace the coth(x) term in the correlation
        function's real part with 1. At higher temperatures the coth(x)
        function behaves poorly at low frequencies.
    
    Returns
    =======
    corrR: ndarray
        A 1D array of the real part of the correlation function
    
    corrI: ndarray
        A 1D array of the imaginary part of the correlation function
    """ 
    corrR = []
    corrI = []
    for t in tlist:
        if temperature == 0:
            integrandR = lambda x: np.multiply(spectrum(x, *params), np.cos(x*t))
        else:
            integrandR = lambda x: np.multiply(spectrum(x, *params), coth(x/(2*temperature))*np.cos(x*t))
        integrandI = lambda x: np.multiply(spectrum(x, *params), np.sin(x*t))
        corrR.append(quad(integrandR, wstart, wend)[0])
        corrI.append(quad(integrandI, wstart, wend)[0])
    return np.real(corrR), np.real(corrI)

def sinusoid_fit(tt, yy):
    """
    Fits a sinusoid (cos) curve to the data. The fitting is very sensitive to the
    initial guess and hence we use the Fourier transform to find the dominant frequency.
    A nice discussion is on StackOverflow
    https://stackoverflow.com/questions/16716302/how-do-i-fit-a-sine-curve-to-my-data-with-pylab-and-numpy
    
    Returns
    =======
    fitfunc: callable
        A callable function of the form f(t) which can be used to
        generate the data according to the fitted function and the
        obtained parameters.
    
    params: ndarray
        A 1D array of the fitted parameters
        [A, w, p, c] according to the fit
        
        A * cos(wt) + c    
    """
    tt = np.array(tt)
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing

    Fyy = abs(np.fft.rfft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq,
                         0., guess_offset])

    def cosfunc(t, A, w, p, c):  return A * (np.cos(w*t + p)) + c
    popt, pcov = curve_fit(cosfunc, tt, yy, p0=guess)
    A, w, p, c = popt
    fitfunc = lambda t: A * (np.cos(w*t + p)) + c
    fitparams = np.array([A, w, p, c])
    return fitfunc, fitparams

def envelope_fit(tt, yy):
    """
    Fits an exponential decay to the the envelope of a signal.
    
    Returns
    =======
    fitfunc: callable
        A callable function of the form f(t) which can be used to
        generate the data according to the fitted function and the
        obtained parameters.
    
    params: ndarray
        A 1D array of the fitted parameters
        [A, w, p, c] according to the fit
        
        A e^(-wt + p) + c
    """
    analytic_signal = hilbert(yy)
    amplitude_envelope = np.abs(analytic_signal)
    
    tt = np.array(tt)
    yy = np.array(amplitude_envelope)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing

    Fyy = abs(np.fft.rfft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq,
                         0., guess_offset])
    
    expfit = lambda x, A, w, p, c: A * np.exp(-w*x + p) + c
    popt, pcov = curve_fit(expfit, xdata=tt, ydata=amplitude_envelope)
    A, w, p, c = popt
    fitfunc = lambda x: A * np.exp(-w*x + p) + c
    fitparams = np.array([A, w, p, c])
    return fitfunc, fitparams

def get_exponents(t, y, num_terms=3):
    """
    Calculates the function y as a sum of exponents by first fitting the envelope
    to an exponential decay and then the remaining part iteratively to a sum of
    cosines so that it is expressed as as sum of terms: A exp(-bt)
    
    Parameters
    ==========
    t: ndarry
        1D array denoting the time.
        
    y: ndarray
        1D array with the signal
        
    num_term: int
        The number of terms to use for fitting
    """
    envelope_func, envelope_params = envelope_fit(t, y)
    residue = y/envelope_func(t)

    xdata, ydata = np.array(t), np.array(residue)
    
    f, p = sinusoid_fit(xdata, ydata)
    res = [(f, p)]
    cosine_params = [p]
    for i in range(0, num_terms):
        f1, p1 = sinusoid_fit(xdata, ydata - res[i][0](xdata))
        res.append((f1, p1))
        cosine_params.append(p1)
        ydata = ydata - res[i][0](xdata)    
    env, sinusoid = np.array(envelope_params), np.array(cosine_params)
    return _exponents(env, sinusoid)

def _exponents(envelope_params, cosine_params):
    """
    Re-packs the parameters from the envelope and the sinusoid fit
    to a list of prefactors (amplitudes) and frequencies. These can
    be obtained from the call:
    >> envelope_params, cosine_params = _get_exponents(t, y)
    
    Parameters
    ==========
    envelope_params: ndarray
        A 1D array with the parameters [A, w, p, c] corresponding to the
        fit of the envelope as `Ae^(wt + p) + c`
    
    cosine_params: ndarray
        A 2D array of shape (num_terms, 4) where each element gives the
        terms [A, w, p, c] for the fitting of a cosine term `A cos(wt + p) + c`
        corresponding to the fitting of the correlation function.

    Returns
    =======
    amplitudes: ndarray
        1D array containing the amplitudes for the fit [A1, A2, ...]
    
    frequencies: ndarray
        1D array containing the frequencies for the fit [b1, b2, ...]
    """
    freq_pos = 1j * cosine_params[:, 1].copy()
    freq_neg = -freq_pos
    amp_pos = np.multiply(np.exp(1j*cosine_params[:, 2]), cosine_params[:, 0])/2
    amp_neg = np.multiply(np.exp(-1j*cosine_params[:, 2]), cosine_params[:, 0])/2
    constant_term = np.sum(cosine_params[:, 3])
    
    # Multiply with the envelope terms
    A0, w0, p0, c0 = envelope_params
    
    f1 = freq_pos + w0
    f2 = freq_neg + w0
    f3 = np.array([w0])
    
    a1 = amp_pos*A0*np.exp(p0)
    a2 = amp_neg*A0*np.exp(p0)
    a3 = np.array([constant_term*A0*np.exp(p0)])
    
    f4 = freq_pos.copy()
    f5 = freq_neg.copy()
    f6 = np.array([0])

    a4 = amp_pos * c0
    a5 = amp_neg * c0
    a6 = np.array([constant_term*c0])

    amplitudes, frequencies = np.concatenate([a1, a2, a3, a4, a5, a6]), -np.concatenate([f1, f2, f3, f4, f5, f6])
    ordered = np.argsort(amplitudes)
    
    return amplitudes[-ordered], frequencies[-ordered]

