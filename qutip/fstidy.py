from fseries import *

def fstidy(fs,*args):
	'''FSTIDY tidies up a function series, removing small terms
	fs = fstidy(fs1,tol)
	fs1 is input function series. Terms with identical parameters are merged.
	Terms are deleted if amplitude is smaller than tol
	tol = [reltol,abstol]. Default tolerance is [1e-6,1e-6].'''
	if not any(args):
		tol=[1e-6,1e-6]
	if len(args)==1:
		args[1]=0
	numterms=fs.nterms()
	ftype=fs.ftype
	fparam=fs.fparam
	data=array([fs.series[k].full().max() for k in range(0,numterms)])
	Amax=data
	return Amax 
	
	
