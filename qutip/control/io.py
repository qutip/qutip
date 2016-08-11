# -*- coding: utf-8 -*-
# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2016 and later, Alexander J G Pitchford
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

import os
import errno

def create_dir(dir_name, desc='output'):
    """
    Checks if the given directory exists, if not it is created
    Returns
    -------
    dir_ok : boolean
        True if directory exists (previously or created)
        False if failed to create the directory

    dir_name : string
        Path to the directory, which may be been made absolute

    msg : string
        Error msg if directory creation failed
    
    """

    dir_ok = True
    if '~' in dir_name:
        dir_name = os.path.expanduser(dir_name)
    elif not os.path.abspath(dir_name):
        # Assume relative path from cwd given
        dir_name = os.path.join(os.getcwd(), dir_name)

    msg = "{} directory is ready".format(desc)
    errmsg = "Failed to create {} directory:\n{}\n".format(desc,
                                                        dir_name)

    if os.path.exists(dir_name):
        if os.path.isfile(dir_name):
            dir_ok = False
            errmsg += "A file already exists with the same name"
    else:
        try:
            os.makedirs(dir_name)
            msg += ("directory {} created "
                        "(recursively)".format(dir_name))
        except OSError as e:
            if e.errno == errno.EEXIST:
                msg += ("Assume directory {} created "
                    "(recursively) by some other process. ".format(dir_name))
            else:
                dir_ok = False
                errmsg += "Underling error (makedirs) :({}) {}".format(
                    type(e).__name__, e)

    if dir_ok:
        return dir_ok, dir_name, msg
    else:
        return dir_ok, dir_name, errmsg
        