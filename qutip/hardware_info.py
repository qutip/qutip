# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
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

__all__ = ['hardware_info']

import os
import sys
import multiprocessing
import numpy as np

def _mac_hardware_info():
    info = dict()
    results = dict()
    for l in [l.split(':') for l in os.popen('sysctl hw').readlines()[1:]]:
        info[l[0].strip(' "').replace(' ', '_').lower().strip('hw.')] = \
            l[1].strip('.\n ')
    results.update({'cpus': int(info['physicalcpu'])})
    results.update({'cpu_freq': int(float(os.popen('sysctl -n machdep.cpu.brand_string')
                                .readlines()[0].split('@')[1][:-4])*1000)})
    results.update({'memsize': int(int(info['memsize']) / (1024 ** 2))})
    # add OS information
    results.update({'os': 'Mac OSX'})
    return results


def _linux_hardware_info():
    results = {}
    # get cpu number
    sockets = 0
    cores_per_socket = 0
    frequency = 0.0
    for l in [l.split(':') for l in open("/proc/cpuinfo").readlines()]:
        if (l[0].strip() == "physical id"):
            sockets = np.maximum(sockets,int(l[1].strip())+1)
        if (l[0].strip() == "cpu cores"):
            cores_per_socket = int(l[1].strip())
        if (l[0].strip() == "cpu MHz"):
            frequency = float(l[1].strip()) / 1000.
    results.update({'cpus': sockets * cores_per_socket})
    # get cpu frequency directly (bypasses freq scaling)
    try:
        file = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"
        line = open(file).readlines()[0]
        frequency = float(line.strip('\n')) / 1000000.
    except:
        pass
    results.update({'cpu_freq': frequency})

    # get total amount of memory
    mem_info = dict()
    for l in [l.split(':') for l in open("/proc/meminfo").readlines()]:
        mem_info[l[0]] = l[1].strip('.\n ').strip('kB')
    results.update({'memsize': int(mem_info['MemTotal']) / 1024})
    # add OS information
    results.update({'os': 'Linux'})
    return results

def _freebsd_hardware_info():
    results = {}
    results.update({'cpus': int(os.popen('sysctl -n hw.ncpu').readlines()[0])})
    results.update({'cpu_freq': int(os.popen('sysctl -n dev.cpu.0.freq').readlines()[0])})
    results.update({'memsize': int(os.popen('sysctl -n hw.realmem').readlines()[0]) / 1024})
    results.update({'os': 'FreeBSD'})
    return results

def _win_hardware_info():
    try:
        from comtypes.client import CoGetObject
        winmgmts_root = CoGetObject("winmgmts:root\cimv2")
        cpus = winmgmts_root.ExecQuery("Select * from Win32_Processor")
        ncpus = 0
        for cpu in cpus:
            ncpus += int(cpu.Properties_['NumberOfCores'].Value)
    except:
        ncpus = int(multiprocessing.cpu_count())
    return {'os': 'Windows', 'cpus': ncpus}


def hardware_info():
    """
    Returns basic hardware information about the computer.

    Gives actual number of CPU's in the machine, even when hyperthreading is
    turned on.

    Returns
    -------
    info : dict
        Dictionary containing cpu and memory information.

    """
    if sys.platform == 'darwin':
        out = _mac_hardware_info()
    elif sys.platform == 'win32':
        out = _win_hardware_info()
    elif sys.platform in ['linux', 'linux2']:
        out = _linux_hardware_info()
    elif sys.platform.startswith('freebsd'):
        out = _freebsd_hardware_info()
    else:
        out = {}
    return out

if __name__ == '__main__':
    print(hardware_info())
