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

from pylab import *
from matplotlib import cm
from istests import *


#
# A collection of various visalization functions.
#


# Adopted from the SciPy Cookbook.
def _blob(x, y, w, w_max, area):
    """
    Draws a square-shaped blob with the given area (< 1) at
    the given coordinates.
    """
    hs = sqrt(area) / 2
    xcorners = array([x - hs, x + hs, x + hs, x - hs])
    ycorners = array([y - hs, y - hs, y + hs, y + hs])

    fill(xcorners, ycorners, color=cm.jet(int((w+w_max) * 256 / (2*w_max))))

# Adopted from the SciPy Cookbook.
def hinton(rho):
    """
    Draws a Hinton diagram for visualizing a density matrix. 
    
    Args:
        
        rho (Qobj) for input density matrix.
    
    Raises:
        
        ValueError if input argument is not a quantum object.
        
    """
    if not isoper(rho):
        raise ValueError("argument must be a quantum operator")

    W = rho.full()

    clf()
    height, width = W.shape
  
    w_max = 1.25 * max(abs(diag(matrix(W))))
    if w_max <= 0.0:
        w_max = 1.0

    fill(array([0,width,width,0]),array([0,0,height,height]), color=cm.jet(128))
    axis('off')
    axis('equal')
    for x in xrange(width):
        for y in xrange(height):
            _x = x+1
            _y = y+1
            if real(W[x,y]) < 0.0:
                _blob(_x - 0.5, height - _y + 0.5,  abs(W[x,y]), w_max, min(1,abs(W[x,y])/w_max))
            else:
                _blob(_x - 0.5, height - _y + 0.5, -abs(W[x,y]), w_max, min(1,abs(W[x,y])/w_max))


