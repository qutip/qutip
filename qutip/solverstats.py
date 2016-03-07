# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2016 and later, Alexander J. G. Pitchford
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

import sys
import datetime
from collections import OrderedDict

def _format_time(t, tt=None, ttt=None):
    time_str = str(datetime.timedelta(seconds=t))
    if tt is not None and ttt is not None:
        sect_percent = 100*t/tt
        solve_percent = 100*t/ttt
        time_str += " ({:03.2f}% section, {:03.2f}% total)".format(
                                            sect_percent, solve_percent)
    elif tt is not None:
        sect_percent = 100*t/tt
        time_str += " ({:03.2f}% section)".format(sect_percent)
        
    elif ttt is not None:
        solve_percent = 100*t/ttt
        time_str += " ({:03.2f}% total)".format(solve_percent)
                                            
    return time_str

class SolverStats(object):
    """
    Statistical information on the solver performance
    Statistics can be grouped into sections.
    If no section names are given in the the contructor, then all statistics
    will be added to one section 'main'
    
    Parameters
    ----------
    section_names : list
        list of keys that will be used as keys for the sections
        These keys will also be used as names for the sections
        The text in the output can be overidden by setting the header property
        of the section
        If no names are given then one section called 'main' is created
        
    Attributes
    ----------
    sections : OrderedDict of _SolverStatsSection
        These are the sections that are created automatically on instantiation
        or added using add_section
        
    header : string
        Some text that will be used as the heading in the report
        By default there is None
        
    total_time : float
        Time in seconds for the solver to complete processing
        Can be None, meaning that total timing percentages will be reported
        
    Methods
    -------
    add_section
        Add another section
        
    add_count
        Add some stat that is an integer count
        
    add_timing
        Add some timing statistics
        
    add_message
        Add some text type for output in the report
        
    report:
        Output the statistics report to console or file.
    """

    def __init__(self, section_names=None):
        self._def_section_name = 'main'
        self.sections = OrderedDict()
        self.total_time = None
        self.header = None
        if isinstance(section_names, list):
            c = 0
            for name in section_names:
                self.sections[name] = _SolverStatsSection(name, self)
                if c == 0:
                    self._def_section_name = name
                c += 1
                
        else:
            self.sections[self._def_section_name] = \
                        _SolverStatsSection(self._def_section_name)
    
    def _get_section(self, section):
        if section is None:
            return self.sections[self._def_section_name]
        elif isinstance(section, _SolverStatsSection):
            return section
        else:
            sect = self.sections.get(section, None)
            if sect is None:
                raise ValueError("Unknown section {}".format(section))
            else:
                return sect
                
    def add_section(self, name):
        """
        Add another section with the given name
        
        Parameters
        ----------
        name : string
            will be used as key for sections dict
            will also be the header for the section
        
        Returns
        -------
        section : `class` : _SolverStatsSection
            The new section
        """
        sect = _SolverStatsSection(name, self)
        self.sections[name] = sect
        return sect
        
    def add_count(self, key, value, section=None):
        """
        Add value to count. If key does not already exist in section then
        it is created with this value.
        If key already exists it is increased by the give value
        value is expected to be an integer
        
        Parameters
        ----------
        key : string
            key for the section.counts dictionary
            reusing a key will result in numerical addition of value
            
        value : int
            Initial value of the count, or added to an existing count
        
        section: string or `class` : _SolverStatsSection
            Section which to add the count to.
            If None given, the default (first) section will be used
        """
                
        self._get_section(section).add_count(key, value)
        
    def add_timing(self, key, value, section=None):
        """
        Add value to timing. If key does not already exist in section then
        it is created with this value.
        If key already exists it is increased by the give value
        value is expected to be a float, and given in seconds.
        
        Parameters
        ----------
        key : string
            key for the section.timings dictionary
            reusing a key will result in numerical addition of value
            
        value : int
            Initial value of the timing, or added to an existing timing
        
        section: string or `class` : _SolverStatsSection
            Section which to add the timing to.
            If None given, the default (first) section will be used
        """               
        self._get_section(section).add_timing(key, value)
            
    def add_message(self, key, value, section=None, sep=";"):
        """
        Add value to message. If key does not already exist in section then
        it is created with this value.
        If key already exists the value is added to the message
        The value will be converted to a string
        
        Parameters
        ----------
        key : string
            key for the section.messages dictionary
            reusing a key will result in concatenation of value
            
        value : int
            Initial value of the message, or added to an existing message
            
        sep : string
            Message will be prefixed with this string when concatenating
        
        section: string or `class` : _SolverStatsSection
            Section which to add the message to.
            If None given, the default (first) section will be used
        """                
        self._get_section(section).add_message(key, value, sep=sep)
    
    def set_total_time(self, value, section=None):
        """
        Sets the total time for the complete solve or for a specific section
        value is expected to be a float, and given in seconds
        
        Parameters
        ----------
        value : float
            Time in seconds to complete the solver section
            
        section : string or `class` : _SolverStatsSection
            Section which to set the total_time for
            If None given, the total_time for complete solve is set
        """
        if not isinstance(value, float):
            try:
                value = float(value)
            except:
                raise TypeError("value is expected to be a float")
        
        if section is None:
            self.total_time = value
        else:
            sect = self._get_section(section)
            sect.total_time = value
                
    def report(self, output=sys.stdout):
        """
        Report the counts, timings and messages from the sections.
        Sections are reported in the order that the names were supplied
        in the constructor.
        The counts, timings and messages are reported in the order that they
        are added to the sections
        The output can be written to anything that supports a write method,
        e.g. a file or the console (default)
        The output is intended to in markdown format
        
        Parameters
        ----------
        output : stream
            file or console stream - anything that support write - where
            the output will be written
        """
        
        if not hasattr(output, 'write'):
            raise TypeError("output must have a write method")
        
        if self.header:
            output.write("{}\n{}\n".format(self.header, 
                                     ("="*len(self.header))))
        for name, sect in self.sections.items():
            sect.report(output)
            
        if self.total_time is not None:
            output.write("\nSummary\n-------\n")
            output.write("{}\t solver total time\n".format(
                                            _format_time(self.total_time)))
            
class _SolverStatsSection(object):
    """
    Not intended to be directly instantiated
    This is the type for the SolverStats.sections values
    
    The method parameter descriptions are the same as for those the parent 
    with the same method name
    
    Parameters
    ----------
    name : string
        key for the parent sections dictionary
        will also be used as the header
    
    parent : `class` :  SolverStats
        The container for all the sections
        
    Attributes
    ----------
    name : string
        key for the parent sections dictionary
        will also be used as the header
    
    parent : `class` :  SolverStats
        The container for all the sections
        
    header : string
        Used as heading for section in report
        
    counts : OrderedDict
        The integer type statistics for the stats section
        
    timings : OrderedDict
        The timing type statistics for the stats section
        Expected to contain float values representing values in seconds
        
    messages : OrderedDict
        Text type output to be reported
    
    total_time : float
        Total time for processing in the section
        Can be None, meaning that section timing percentages will be reported
    """
    def __init__(self, name, parent):
        self.parent = parent
        self.header = str(name)
        self.name = name
        self.counts = OrderedDict()
        self.timings = OrderedDict()
        self.messages = OrderedDict()
        self.total_time = None

    def add_count(self, key, value):
        """
        Add value to count. If key does not already exist in section then
        it is created with this value.
        If key already exists it is increased by the given value
        value is expected to be an integer
        """
        if not isinstance(value, int):
            try:
                value = int(value)
            except:
                raise TypeError("value is expected to be an integer")
                
        if key in self.counts:
            self.counts[key] += value
        else:
            self.counts[key] = value
        
    def add_timing(self, key, value):
        """
        Add value to timing. If key does not already exist in section then
        it is created with this value.
        If key already exists it is increased by the give value
        value is expected to be a float, and given in seconds.
        """
        if not isinstance(value, float):
            try:
                value = float(value)
            except:
                raise TypeError("value is expected to be a float")
                
        if key in self.timings:
            self.timings[key] += value
        else:
            self.timings[key] = value
            
    def add_message(self, key, value, sep=";"):
        """
        Add value to message. If key does not already exist in section then
        it is created with this value.
        If key already exists the value is added to the message
        The value will be converted to a string
        """
        value = str(value)

        if key in self.messages:
            if sep is not None:
                try:
                    value = sep + value
                except:
                    TypeError("It is not possible to concatenate the value with "
                                "the given seperator")
            self.messages[key] += value
        else:
            self.messages[key] = value
            
    def report(self, output=sys.stdout):
        """
        Report the counts, timings and messages for this section.
        Note the percentage of the section and solver total times will be
        given if the parent and or section total_time is set
        """
        if self.header:
            output.write("\n{}\n{}\n".format(self.header, 
                                     ("-"*len(self.header))))
        
        # TODO: Make the timings and counts ouput in a table format
        #       Generally make more pretty
        
        # Report timings
        try:
            ttt = self.parent.total_time
        except:
            ttt = None
            
        tt = self.total_time
        
        output.write("### Timings:\n")
        for key, value in self.timings.items():
            l = " - {}\t{}\n".format(_format_time(value, tt, ttt), key)
            output.write(l)
        if tt is not None:
            output.write(" - {}\t{} total time\n".format(_format_time(tt), 
                                                     self.name))
            
        # Report counts
        output.write("### Counts:\n")
        for key, value in self.counts.items():
            l = " - {}\t{}\n".format(value, key)
            output.write(l)
        
        # Report messages
        output.write("### Messages:\n")
        for key, value in self.messages.items():
            l = " - {}:\t{}\n".format(key, value)
            output.write(l)
            
