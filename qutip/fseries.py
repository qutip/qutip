#This file is part of QuTIP.
#
#    QuTIP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#    QuTIP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with QuTIP.  If not, see <http://www.gnu.org/licenses/>.
###########################################################################

from scipy import array
import scipy.linalg as la
import scipy.sparse as sp
from qobj import *

class fseries:
	"""
    fseries object class.
    requires: scipy, scipy.sparse.csr_matrix, scipy.linalg
    """
    ################## Define fseries class #################
	def __init__(self,q=array([0]),ftype=[[]],fparam=[[]]):
		if (not any(q)) & (not any(ftype)) & (not any(fparam)):#no input params
			qo=qobj() #create blank qobj
			self.series=array([qo])
			self.dims=qo.dims
			self.shape=qo.shape
			self.ftype=[[]]
			self.fparam=[[]]
		elif any(q):
			if isinstance(q,fseries):
				self.series=q.series
				self.dims=q.dims
				self.shape=q.shape
				self.ftype=q.ftype
				self.fparam=q.fparam 
			elif (not isinstance(q,fseries)) & (not any(ftype)) & (not any(fparam)):
				qo=qobj(q)
				self.series=array([qo])
				self.dims=qo.dims
				self.shape=qo.shape
			self.ftype=[[0]]
			self.fparam=[[]]
			if (not isinstance(q,fseries)) & any(ftype) & (not any(fparam)):
				if len(ftype)!=1:
					raise TypeError('Invalid function type')
				qo=qobj(q)
				self.series=array([qo])
				self.dims=qo.dims
				self.shape=qo.shape
				self.ftype=ftype
				self.fparam=[[]]
			elif (not isinstance(q,fseries)) & any(ftype) & (any(fparam)):
				qo=qobj(q)
				self.series=array([qo])
				self.dims=qo.dims
				self.shape=qo.shape
				self.ftype=ftype
				self.fparam=fparam
		nampl=1
		if (nampl!=len(self.fparam)) or (nampl!=len(self.ftype)):
			raise ValueError('Amplitude object has incorrect number of members')
		
	def __str__(self):
		nterms=self.nterms()
		print "FSERIES object: " +"function series has: "+str(self.nterms())+" term(s)"+ ", ftype = " + str(self.ftype) + ", fparam = " + str(self.fparam) + "\n Hilbert Space size = "+ str(self.series[0].shape)+"\n"
		for k in range(0,nterms):
			print "function #",k,':'
			print self.series[k].full()
			print '\n'
		return ''
	def __add__(self,other):
		#defines addition for FSERIES objects
		fself=fseries(self)
		fother=fseries(other)
		if fself.dims==fother.dims:
			if len(fself.ftype[0])==1:
				out=fseries()
				out.series=[fself.series[0],other.series[0]]
				out.dims=fself.dims
				out.shape=fself.shape
				out.ftype=[[fself.ftype[0][0],fother.ftype[0][0]]]
				out.fparam=[fself.fparam,fother.fparam]
			else:
				out=fseries()
				out.series=[fself.series[0]+fself.series[1],fother.series[0]]
				out.dims=fself.dims
				out.shape=fself.shape
				out.ftype=[[fself.ftype[0][0],fother.ftype[0][0]]]
				out.fparam=[fself.fparam,fother.fparam]
			return out
		else:
			raise TypeError('Incompatible Hilbert Space dimensions')
			
	def __sub__(self,other):#self-other
		#defines substraction for FSERIES objects
		return self+(-other)
	def __rsub__(self,other):#other-self
		#defines right subtraction
		return other+(-self)
		
	def __mul__(self,other):
		#defines multiplication with qobj on left (ex. qobj*5)
		if isinstance(other,qobj) or isinstance(other,(int,float,complex)):
			qother=qobj(other)
			d=dimprod(qother.dims,self.dims)
			fsshape=dims2shape(d)
			ftype=self.ftype
			fparam=self.fparam
			shape2=self.shape
			if shape2[0]!=self.shape[1]:
				shape2=[prod(self.shape),1]
			
			
		
	def __neg__(self):
		out=fseries()
		out.series=-self.series
		out.dims=self.dims
		out.shape=self.shape
		out.ftype=self.ftype
		out.fparam=self.fparam
		return out
		
	def nterms(self):
		return len(self.series)
		
			

################---supplimentary functions---###############
def dimprod(d1,d2):
	if prod(d1[0])==1 & prod(d2[1])==1:
		d=d2
		return d
	if prod(d2[0])==1 & prod(d1[1])==1:
		d=d1
		return d
	if d1[1]==d2[0]:
		return [d1[0],d2[1]]
	elif d1[1]==d2:
		return d1[0]
	else:
		raise TypeError('Incompatible matrices in product') 


def dims2shape(qdims):
	return [prod(qdims[0]),prod(qdims[1])]

















