'''
Code for simulating secure key rate, twofolds, and quantum bit error rate
Written in Python and QuTIP by Catherine Holloway (c2hollow@iqc.ca).

Detector model and squashing functions by Catherine Holloway,
based on code by Dr. Thomas Jennewein (tjennewe@iqc.ca).

Contributed to the QuTiP project on June 06, 2012 by Catherine Holloway.
'''

#imports
from qutip import *
from numpy import *
from pylab import *
import matplotlib
import matplotlib.pyplot as plt


def choose(n, k):
	"""
	Binomial coefficient function for the detector model.
	
	Parameters
	----------
	n : int
	    Number of elements.
	k : int
	    Number of subelements.
	
	Returns
	-------
	coeff : int
	    Binomial coefficient.
	
	"""
	if 0 <= k <= n:
		ntok = 1
		ktok = 1
		for t in xrange(1, min(k, n - k) + 1):
			ntok *= n
			ktok *= t
			n -= 1
		return ntok // ktok
	else:
		return 0


def BucketDetector_realistic_detector(N,efficiency,n_factor):
	"""
	Bucket detector model based on H. Lee, U. Yurtsever, P. Kok, G. Hockney, C. Adami, S. Braunstein,
	and J. Dowling, "Towards photostatistics from photon-number discriminating detectors,"
	Journal of Modern Optics, vol. 51, p. 15171528, 2004.
	
	Parameters
	----------
	N : int 
	    The Fock Space dimension.
	efficiency : float
	    The channel efficiency.
	n_factor : float
	    The average number of dark counts per detection window APD (Bucket Detector).
	
	Returns
	-------
	[proj, un_proj] : list
	    The projection and unprojection operators.
	
	"""
	proj=zeros((N,N))
	#APD (Bucket Detector) un_detector (=gives probability for 0-detection)
	un_proj=identity(N)
	#n_factor = 0;
	for i in range(N):
	    probs = 0;
	    for k in range (1,100):
	        for d in range(k+1):
	            if k-d<=i:
	                probs= probs+ (exp(-n_factor)*(n_factor)**(d))/factorial(d)*choose(i,k-d)*efficiency**(k-d)*(1-efficiency)**(i-k+d)
	        
	    proj[i,i]=probs
	   
	
	un_proj = un_proj-proj
	un_proj = Qobj(un_proj)
	proj = Qobj(proj)
	return [proj,un_proj]


def measure_2folds_4modes_squashing(N,psi,proj,proj2):
	"""
	Determines the 2-fold count rate on the joint state 
	outputs for an array of double count probabilities.
	
	Parameters
	----------
	N : int
	    The Fock Space dimension.
	psi : qobj
	    The entangled state to analyze
	proj1 : qobj
	    1st projection operator for the Channel between Alice and
    	the Channel between Bob.
	proj2 : qobj
	    2nd projection operator for the Channel between Alice and 
	    the Channel between Bob.
	
	Returns
	-------
	[HH,HV,VH,VV] : list
	    Two-fold probabilities.
	
	Notes
	-----
	The squashing (assigning double pairs to random bases) comes from two papers:
	
	    T. Moroder, O. Guhne, N. Beaudry, M. Piani, and N. Lutkenhaus,
	    "Entanglement verication with realistic measurement devices via squashing operations,"
	    Phys. Rev. A, vol. 81, p. 052342, May 2010.
	
	    N. Lutkenhaus, "Estimates for practical quantum cryptography," Phys. Rev.A,
	    vol. 59, pp. 3301-3319, May 1999.
	
	"""
	ida=qeye(N)
	final_state=psi
	det_exp = zeros((2,2,2,2))

	#i,j,k,l means Ha,Va,Hb,Vb, 0 means detector clicked, 1 means detector did not click
	for i in range(2):
		for j in range(2):
			for k in range(2):
				for l in range(2):
					#expectation values for different detector configurations
					det_exp[i][j][k][l] = abs(expect(tensor(proj[i],proj[j],proj2[k],proj[l]),final_state))
	#two fold probabilities
	HH = det_exp[0][1][0][1]+0.5*(det_exp[0][0][0][1]+det_exp[0][1][0][0])+0.25*det_exp[0][0][0][0]
	VV = det_exp[1][0][1][0]+0.5*(det_exp[0][0][1][0]+det_exp[1][0][0][0])+0.25*det_exp[0][0][0][0]
	HV = det_exp[0][1][1][0]+0.5*(det_exp[0][0][1][0]+det_exp[0][1][0][0])+0.25*det_exp[0][0][0][0]
	VH = det_exp[1][0][0][1]+0.5*(det_exp[0][0][0][1]+det_exp[1][0][0][0])+0.25*det_exp[0][0][0][0]

	return [HH,HV,VH,VV]


def sim_qkd_entanglement(eps,loss_a,loss_b,n_factor_a,n_factor_b,N):
	"""
	Simulate skr with an SPDC state.
	
	Parameters
	----------
	eps : float
	    The squeezing factor, sort of analogous to the amount of 
	    pumping power to the spdc source, but not really.
	loss_a : float
	    Efficiency of the quantum channel going to Alice.
	loss_b : float
	    Efficiency of the quantum channel going to Bob. 
	n_factor_a : float
	    Background noise in Alice's detection.
	n_factor_b : float
	    Background noise in Bob's detection.
	N : int
	    Size of the fock space that we allow for the states
	
	Returns
	-------
	qber : float
	    The Quantum Bit Error Rate
	twofolds : float
	    Probability of Alice and Bob getting a simultaneous detection 
	    of a photon pair (also referred to as coincidences) within a 
	    timing window.
	skr : float
	    Probability of getting a secure key bit within a timing window, 
	    assuming error correction and privacy amplification, in the 
	    limit of many coincidences.
    
    """
	#make vaccuum state
	vacc = basis(N,0)

	#make squeezing operator for SPDC
	H_sq = 1j*eps*(tensor(create(N),create(N))+tensor(destroy(N),destroy(N)))
	
	#exponentiate hamiltonian and apply it to vaccuum state to make an SPDC state
	U_sq = H_sq.expm()
	spdc = U_sq*tensor(vacc,vacc)
	psi = tensor(spdc,spdc)
	#since qutip doesn't have a permute function, 
	#we have to do a couple of steps in between
	#1. turn psi from a sparse matrix to a full matrix
	out = psi.full()
	#2. reshape psi into a 4-D matrix
	out = reshape(out, (N,N,N,-1))
	#3. permute the dimensions of our 4-D matrix
	out = transpose(out,(0,3,2,1))
	#4. turn the matrix back into a 1-D array 
	out = reshape(out,(N*N*N*N,-1))
	#5. convert the matrix back into a quantum object
	psi = Qobj(out,dims = [[N, N, N, N], [1, 1, 1, 1]])

	# model detectors
	a_det = BucketDetector_realistic_detector(N,loss_a,n_factor_a)
	b_det = BucketDetector_realistic_detector(N,loss_b,n_factor_b)
	
	#measure detection probabilities
	probs2f=measure_2folds_4modes_squashing(N,psi,a_det,b_det)

	#Rates returned are 'per pulse', so multiply by source rate
	twofolds=probs2f[0]+probs2f[1]+probs2f[2]+probs2f[3]
	#Determine QBER from returned detection probabilities
	qber = (probs2f[0]+probs2f[3])/twofolds

	#calculate the entropy of the qber  
	if qber>0:
		H2=-qber*log2(qber) - (1-qber)*log2(1-qber)
	else:
		H2 = 0
	# estimate error correction efficiency from the CASCADE algorithm 
	f_e = 1.16904371810274 + qber
	#security analysis - calculate skr in infinite key limit
	#See Chris Erven's PhD thesis or Xiongfeng Ma's paper 
	#to understand where this equation comes from
	skr=real(twofolds*0.5*(1-(1+f_e)*H2))
	return [qber, skr, twofolds]


if __name__=='__main__':
	#Lets look at what happens to the secure key rate and 
	#the quantum bit error rate as the loss gets worse.
	#Analogous to distance with fiber optic links.
	
	#define the fock space
	N = 7
	#define the squeezing paramter
	eps = 0.2
	#define the noise factor
	n_factor = 4.0e-5
	#define the length of the coincidence window (in s)
	coinc_window = 2.0e-9
	loss_db = arange(0,30)
	skr = zeros(30)
	qber = zeros(30)
	twofolds = zeros(30)
    
    #run calculation
	for i in range(30):
		exp_loss = 10.0**(-loss_db[i]/10.0);
		[qber[i], skr[i], twofolds[i]] = sim_qkd_entanglement(eps,exp_loss,exp_loss,n_factor,n_factor,N)
	skr = skr/coinc_window
	qber = qber*100
    
    #plot results
	fig = plt.figure()
	ax = fig.add_subplot(211)
	ax.plot(loss_db, skr,lw=2)
	ax.set_yscale('log')
	ax.set_ylabel('Secure Key Rate (bits/s)')
	ax.set_xlabel('Loss (dB)')
	ax = fig.add_subplot(212)
	ax.plot(loss_db, qber,lw=2)
	ax.set_ylabel('Quantum Bit Error Rate (%)')
	ax.set_ylim([0,15])
	ax.set_xlabel('Loss (dB)')
	plt.show()

