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


from qutip import *
from numpy import allclose
from numpy.testing import assert_equal


def test_jmat_12():
    "Spin 1/2 operators"
    spinhalf = jmat(1/2.)
    
    paulix=array([[ 0.0+0.j,  0.5+0.j],[ 0.5+0.j,  0.0+0.j]])
    pauliy=array([[ 0.+0.j ,  0.+0.5j],[ 0.-0.5j,  0.+0.j ]])
    pauliz=array([[ 0.5+0.j,  0.0+0.j],[ 0.0+0.j, -0.5+0.j]])
    sigmap=array([[ 0.+0.j,  1.+0.j],[ 0.+0.j,  0.+0.j]])
    sigmam=array([[ 0.+0.j,  0.+0.j],[ 1.+0.j,  0.+0.j]])
    
    assert_equal(allclose(spinhalf[0].full(),paulix),True)
    assert_equal(allclose(spinhalf[1].full(),pauliy),True)
    assert_equal(allclose(spinhalf[2].full(),pauliz),True)
    assert_equal(allclose(jmat(1/2.,'+').full(),sigmap),True)
    assert_equal(allclose(jmat(1/2.,'-').full(),sigmam),True)
    
def test_jmat_32():
    "Spin 3/2 operators"
    spin32=jmat(3/2.)
    
    paulix32=array([[ 0.0000000+0.j,  0.8660254+0.j,  0.0000000+0.j,  0.0000000+0.j],
           [ 0.8660254+0.j,  0.0000000+0.j,  1.0000000+0.j,  0.0000000+0.j],
           [ 0.0000000+0.j,  1.0000000+0.j,  0.0000000+0.j,  0.8660254+0.j],
           [ 0.0000000+0.j,  0.0000000+0.j,  0.8660254+0.j,  0.0000000+0.j]])
    
    pauliy32=array([[ 0.+0.j       ,  0.+0.8660254j,  0.+0.j       ,  0.+0.j       ],
           [ 0.-0.8660254j,  0.+0.j       ,  0.+1.j       ,  0.+0.j       ],
           [ 0.+0.j       ,  0.-1.j       ,  0.+0.j       ,  0.+0.8660254j],
           [ 0.+0.j       ,  0.+0.j       ,  0.-0.8660254j,  0.+0.j       ]])
    
    pauliz32=array([[ 1.5+0.j,  0.0+0.j,  0.0+0.j,  0.0+0.j],
           [ 0.0+0.j,  0.5+0.j,  0.0+0.j,  0.0+0.j],
           [ 0.0+0.j,  0.0+0.j, -0.5+0.j,  0.0+0.j],
           [ 0.0+0.j,  0.0+0.j,  0.0+0.j, -1.5+0.j]])
    
    assert_equal(allclose(spin32[0].full(),paulix32),True)
    assert_equal(allclose(spin32[1].full(),pauliy32),True)
    assert_equal(allclose(spin32[2].full(),pauliz32),True)

def test_jmat_42():
    "Spin 2 operators"
    spin42 = jmat(4/2.,'+')
    assert_equal(spin42.dims==[[5],[5]],True)

def test_jmat_52():
    "Spin 5/2 operators"
    spin52 = jmat(5/2.,'+')
    assert_equal(spin52.shape==[6,6],True)
    
def test_destroy():
    "Destruction operator"
    b4=basis(5,4)
    d5=destroy(5)
    test1=d5*b4
    assert_equal(allclose(test1.full(),2.0*basis(5,3).full()), True)
    d3=destroy(3)
    matrix3=array([[ 0.00000000+0.j,  1.00000000+0.j,  0.00000000+0.j],
           [ 0.00000000+0.j,  0.00000000+0.j,  1.41421356+0.j],
           [ 0.00000000+0.j,  0.00000000+0.j,  0.00000000+0.j]])
    
    assert_equal(allclose(matrix3,d3.full()),True)

def test_create():
    "Creation operator"
    b3=basis(5,3)
    c5=create(5)
    test1=c5*b3
    assert_equal(allclose(test1.full(),2.0*basis(5,4).full()), True)
    c3=create(3)
    matrix3=array([[ 0.00000000+0.j,  0.00000000+0.j,  0.00000000+0.j],
           [ 1.00000000+0.j,  0.00000000+0.j,  0.00000000+0.j],
           [ 0.00000000+0.j,  1.41421356+0.j,  0.00000000+0.j]])
    
    assert_equal(allclose(matrix3,c3.full()),True)
    
    
def test_qeye():
    "Identity operator"
    eye3=qeye(5)
    assert_equal(allclose(eye3.full(),eye(5,dtype=complex)),True)
    
def test_num():
    "Number operator"
    n5=num(5)
    assert_equal(allclose(n5.full(),diag([0+0j,1+0j,2+0j,3+0j,4+0j])),True)

def test_squeez():
    "Squeezing operator"
    sq=squeez(4,0.1+0.1j)
    sqmatrix=array([[ 0.99500417+0.j        ,  0.00000000+0.j        ,
             0.07059289-0.07059289j,  0.00000000+0.j        ],
           [ 0.00000000+0.j        ,  0.98503746+0.j        ,
             0.00000000+0.j        ,  0.12186303-0.12186303j],
           [-0.07059289-0.07059289j,  0.00000000+0.j        ,
             0.99500417+0.j        ,  0.00000000+0.j        ],
           [ 0.00000000+0.j        , -0.12186303-0.12186303j,
             0.00000000+0.j        ,  0.98503746+0.j        ]])
             
    assert_equal(allclose(sq.full(),sqmatrix),True)
    
def test_displace():
    "Displacement operator"
    dp=displace(4,0.25)
    dpmatrix=array([[ 0.96923323+0.j, -0.24230859+0.j,  0.04282883+0.j, -0.00626025+0.j],
           [ 0.24230859+0.j,  0.90866411+0.j, -0.33183303+0.j,  0.07418172+0.j],
           [ 0.04282883+0.j,  0.33183303+0.j,  0.84809499+0.j, -0.41083747+0.j],
           [ 0.00626025+0.j,  0.07418172+0.j,  0.41083747+0.j,  0.90866411+0.j]])
    
    
    assert_equal(allclose(dp.full(),dpmatrix),True)
        

