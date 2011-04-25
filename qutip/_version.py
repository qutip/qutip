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
<<<<<<< .mine
import os,re

#set main version here
VERISON=str(0.1)


PATH=os.path.dirname(__file__)
entries_path = PATH+'/.svn/entries'
entries = open(entries_path, 'r').read()
if re.match('(\d+)', entries):
    rev_match = re.search('\d+\s+dir\s+(\d+)', entries)
    if rev_match:
        rev = rev_match.groups()[0]
        __version__=VERISON+" ("+str(rev)+")"
    else:
        __version__=VERISON
else:
    __version__=VERISON=======
import os,re

PATH=os.path.dirname(__file__)
entries_path = PATH+'/.svn/entries'
entries = open(entries_path, 'r').read()
if re.match('(\d+)', entries):
    rev_match = re.search('\d+\s+dir\s+(\d+)', entries)
    if rev_match:
        rev = rev_match.groups()[0]
        __version__="0.1"+" ("+str(rev)+")"
    else:
        __version__="0.1"
else:
    __version__="0.1">>>>>>> .r46
