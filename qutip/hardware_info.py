import os, sys

def _mac_hardware_info():
	info=dict()
	results = dict()
	for l in [l.split(':') for l in os.popen('sysctl hw').readlines()[1:20]]:
		info[l[0].strip(' "').replace(' ', '_').lower().strip('hw.')] = l[1].strip('.\n ')
	results.update({'cpus': int(info['physicalcpu'])})
	results.update({'cpu_freq': int(info['cpufrequency'])/(1000.**3)})
	results.update({'memsize':int(info['memsize'])/(1024**2)})
	return results


def _linux_hardware_info():
	results={}
	#get cpu number
	cpu_info = dict()
	for l in [l.split(':') for l in os.popen('lscpu').readlines()]:
		cpu_info[l[0]] = l[1].strip('.\n ').strip('kB')
	sockets=int(cpu_info['Socket(s)'])
	cores_per_socket=int(cpu_info['Core(s) per socket'])
	results.update({'cpus':sockets*cores_per_socket})
	#get cpu frequency directly (bypasses freq scaling)
	cpu_freq=os.popen('cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq').readlines()[0]
	cpu_freq=float(cpu_freq.strip('\n'))
	results.update({'cpu_freq':cpu_freq/(1000.**2)})
	#get total amount of memory
	mem_info = dict()
	for l in [l.split(':') for l in os.popen('cat /proc/meminfo').readlines()]:
		mem_info[l[0]] = l[1].strip('.\n ').strip('kB')
	results.update({'memsize':int(mem_info['MemTotal'])/1024})
	return results

def _win_hardware_info():
	return None

def hardware_info():
    """
    Returns basic hardware information about the computer.
    
    Returns actual number of CPU's in the machine, even
    when hyperthreading is turned on.
    
    Returns
    -------
    info : dict
        Dictionary containing cpu and memory informaton.
    
    """
    if sys.platform == 'darwin':
        return _mac_hardware_info()
    elif sys.platform == 'win32':
        return _win_hardware_info()
    elif sys.platform == 'linux2':
        return _linux_hardware_info()
    else:
        return None

if __name__=='__main__':
    print hardware_info()
