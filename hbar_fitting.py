import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d as gauss1D
from scipy import signal
from scipy.optimize import curve_fit

def Lorentz(x,x0,w,A,B):
    return A*(1-1/(1+((x-x0)/w)**2))+B

def Cos(x, a, f, os):
    return os + a * np.cos(2 * np.pi * (f * x ))

def Exp_decay(x, A, tau, ofs):
    return A * np.exp(-x / tau) + ofs

def Exp_sine(x, a, tau, ofs, freq , phase):
    return ofs + a * (np.exp(-x/ tau) * np.cos(2 * np.pi * (freq * x + phase)))

def Exp_plus_sine(x, a0, a1,tau1,tau2, ofs, freq , phase):
#     print(a, tau, ofs, freq, phase)
    return ofs + a0 * np.exp(-x/ tau1)*(np.cos(2 * np.pi * (freq * x + phase)))\
        +a1*np.exp(-x/ tau2)

def continue_fourier_transform(time_array,amp_array,freq_array):
    ft_list=[]
    dt=time_array[1]-time_array[0]
    num=len(time_array)
    for w in freq_array:
        phase=np.power((np.zeros(num)+np.exp(-2*np.pi*1j*w*dt)),np.array(range(num)))      
        ft_data=np.sum(amp_array*phase)
        ft_list.append(np.abs(ft_data))
    return np.array(ft_list)


class fitter(object):
    def __init__(self,t_list,amp_list):
        self.t_list=t_list
        self.apm_list=amp_list
    def fit_T1(self):
        x_array=self.t_list
        y_array=self.apm_list
        minimum_amp=np.min(y_array)
        normalization=y_array[-1]
        popt,pcov =curve_fit(Exp_decay,x_array,y_array,[-(normalization-minimum_amp),20,normalization])
        fig,ax=plt.subplots()
        plt.plot(x_array,y_array,label='simulated')
        plt.plot(x_array,Exp_decay(x_array,*popt),label='fitted')
        plt.title(('T1 = %.3f us '% (popt[1])))
        plt.legend()
        plt.show()
        return popt[1]
    
    def fit_phonon_rabi(self):
        x_array=self.t_list
        y_array=self.apm_list
        minimum_point=signal.argrelextrema(y_array, np.less)[0]
        delay_range=x_array[-1]-x_array[0]
        minimum_amp=np.min(y_array)
        max_amp=np.max(y_array)
        freq_guess=1/(x_array[minimum_point[1]]-x_array[minimum_point[0]])

        popt,pcov =curve_fit(Exp_plus_sine,x_array,y_array,[-(max_amp-minimum_amp),0.5,
                                                            delay_range/3,delay_range/3,0,freq_guess,0])
     
        plt.plot(x_array,y_array,label='simulated')
        plt.plot(x_array,Exp_plus_sine(x_array,*popt),label='fitted')
        plt.legend()
        plt.title(' = %.3f us '% (1/popt[-2]/2))
        plt.show()
        return 1/popt[-2]/2