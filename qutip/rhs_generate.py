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
# Copyright (C) 2011-2012, Paul D. Nation & Robert J. Johansson
#
###########################################################################
from cyQ.codegen import Codegen
import odeconfig,os,platform

def rhs_generate(H,H_args,name=None):
    if not name:
        name='rhs'+str(odeconfig.cgen_num)
    try:
        import pyximport
    except:
        raise ImportError("Cython v0.14+ must be installed to run rhs_generate.")
    print "Compiling '"+name+".pyx' ..."
    os.environ['CFLAGS'] = '-w'
    import numpy
    cgen=Codegen(len(H[0]),H[1],H_args)
    cgen.generate(name+".pyx")
    if platform.uname()[0] == 'Windows':
        pyximport.install(setup_args={'include_dirs':[numpy.get_include()],'options': { 'build_ext': { 'compiler': 'mingw32' } }})
    else:
        pyximport.install(setup_args={'include_dirs':[numpy.get_include()]})
    code = compile('from '+name+' import cyq_td_ode_rhs', '<string>', 'exec')
    exec(code)
    odeconfig.tdfunc=cyq_td_ode_rhs
    os.remove(name+".pyx")
    print 'Done.'