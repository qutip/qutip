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
#
# Copyright (C) 2011, Paul D. Nation & Robert J. Johansson
#
###########################################################################
from scipy import *
def sphereplot(theta,phi,values):
	from matplotlib import pyplot, mpl
	from pylab import plot,show,meshgrid,figure
	from mpl_toolkits.mplot3d import Axes3D
	thetam,phim = meshgrid(theta, phi)
	xx=sin(thetam)*cos(phim)
	yy=sin(thetam)*sin(phim)
	zz=cos(thetam)
	r=array(abs(values))
	ph=angle(values)
	nrm=mpl.colors.Normalize(ph.min(),ph.max())
	fig = figure()
	ax =Axes3D(fig)
	surf=ax.plot_surface(r*xx,r*yy,r*zz,rstride=1, cstride=1,facecolors=cm.jet(nrm(ph)),linewidth=0)
	cax,kw=mpl.colorbar.make_axes(ax,shrink=.66,pad=.02)
	cb1=mpl.colorbar.ColorbarBase(cax,cmap=cm.jet,norm=nrm)
	cb1.set_label('Angle')
	show()
	return 



	