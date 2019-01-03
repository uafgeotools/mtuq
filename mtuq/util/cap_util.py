

import csv
import numpy as np
import warnings
import obspy
from copy import deepcopy
from mtuq.util.wavelets import Wavelet

#
# the following functions allow reading in and performing other operations with
# CAP-style weight files. Such files can be used to control the weight on  
# individual stations and components in a moment tensor inversion
#

def parse_weight_file(filename):
    """ Parses CAP-style weight file
    """
    weights = {}
    with open(filename) as file:
        reader = csv.reader(file, delimiter=' ', skipinitialspace=True)
        for row in reader:
            id = '.'.join(row[0].split('.')[1:4])
            weights[id] = [float(w) for w in row[1:]]

    return weights


def remove_unused_stations(dataset, filename):
    """ Removes any stations not listed in CAP weight file or any stations
        with all zero weights
    """
    weights = parse_weight_file(filename)

    unused = []
    for stream in dataset:
        id = stream.id
        if id not in weights:
             unused+=[id]
             continue

        if weights[id][1]==weights[id][2]==\
           weights[id][3]==weights[id][4]==weights[id][5]==0.:
             unused+=[id]

    for id in unused:
        dataset.remove(id)

#
# These rupture and rise time utilties can be used to generate source-time
# functions which match those defined in the cap Perl wrapper.  For use in
# conjunction with mtuq/util/wavelets.py:Trapezoid
#

def cap_rupture_time(Mw):
    if Mw < 1.:
        return 1.
    elif 1. <= Mw <= 9.:
        return int(10.**(0.5*Mw - 2.5) + 0.5)
    elif 9. < Mw:
        return 9.


def cap_rise_time(Mw):
    return 0.5*cap_rupture_time(Mw)


def Nm_to_dynecm(mt):
    """ Converts from N-m (used by MTUQ) to dyne-cm (used by CAP)
    """
    raise NotImplementedError


def _seismic_moment(mt):
    return np.sqrt(np.sum(mt[0:3]**2.) + 0.5*np.sum(mt[3:6]**2.))
 

def _moment_magnitude(mt):
    M0 = _seismic_moment(mt)


#
# the following functions help reproduce cap signal processing functionality.
# See also mtuq/process_data/cap.py
#

def taper(array, taper_fraction=0.3, inplace=True):
    """ Reproduces CAP taper behavior. Similar to obspy Tukey?
    """
    if inplace:
        array = array
    else:
        array = np.copy(array)
    f = taper_fraction
    M = int(round(f*len(array)))
    I = np.linspace(0.,1.,M)
    taper = 0.5*(1-np.cos(np.pi*I))
    array[:M] *= taper
    array[-1:-M-1:-1] *= taper
    if not inplace:
        return array


#
# the following functions are used in benchmark comparisons between CAP and 
# MTUQ synthetics. See tests/benchmark_cap_fk.py
#

def get_synthetics_cap(data, path, event_name):
    container = deepcopy(data)

    for stream in container['body_waves']:
        for trace in stream:
            trace.weight = 1.
            component = trace.meta.channel[-1].upper()

            if component == 'Z':
                filename = '%s/%s.%s.BH.%d' % (path, event_name, stream.id, 9)
                trace_cap = obspy.read(filename, format='sac')[0]

            elif component == 'R':
                filename = '%s/%s.%s.BH.%d' % (path, event_name, stream.id, 7)
                trace_cap = obspy.read(filename, format='sac')[0]

            else:
                continue

            if trace.meta.npts == trace_cap.meta.npts:
                trace.data = trace_cap.data
                # convert from cm/s to m/s
                trace.data *= 1.e-2

            else:
                stream.remove(trace)
            
    for stream in container['surface_waves']:
        for trace in stream:
            trace.weight = 1.
            component = trace.meta.channel[-1].upper()

            if component == 'Z':
                filename = '%s/%s.%s.BH.%d' % (path, event_name, stream.id, 5)
                trace.data = obspy.read(filename, format='sac')[0].data

            if component == 'R':
                filename = '%s/%s.%s.BH.%d' % (path, event_name, stream.id, 3)
                trace.data = obspy.read(filename, format='sac')[0].data

            if component == 'T':
                filename = '%s/%s.%s.BH.%d' % (path, event_name, stream.id, 1)
                trace.data = obspy.read(filename, format='sac')[0].data

            # convert from cm/s to m/s
            trace.data *= 1.e-2

    return container


def get_data_cap(data, path, event_name):
    container = deepcopy(data)

    for stream in container['body_waves']:
        for trace in stream:
            trace.weight = 1.
            component = trace.meta.channel[-1].upper()

            if component == 'Z':
                filename = '%s/%s.%s.BH.%d' % (path, event_name, stream.id, 8)
                trace_cap = obspy.read(filename, format='sac')[0]

            elif component == 'R':
                filename = '%s/%s.%s.BH.%d' % (path, event_name, stream.id, 6)
                trace_cap = obspy.read(filename, format='sac')[0]

            else:
                continue

            if trace.meta.npts == trace_cap.meta.npts:
                trace.data = trace_cap.data
                # convert from cm/s to m/s
                trace.data *= 1.e-2
            else:
                stream.remove(trace)

    for stream in container['surface_waves']:
        for trace in stream:
            trace.weight = 1.
            component = trace.meta.channel[-1].upper()

            if component == 'Z':
                filename = '%s/%s.%s.BH.%d' % (path, event_name, stream.id, 4)
                trace.data = obspy.read(filename, format='sac')[0].data

            elif component == 'R':
                filename = '%s/%s.%s.BH.%d' % (path, event_name, stream.id, 2)
                trace.data = obspy.read(filename, format='sac')[0].data

            elif component == 'T':
                filename = '%s/%s.%s.BH.%d' % (path, event_name, stream.id, 0)
                trace.data = obspy.read(filename, format='sac')[0].data

            # convert from cm/s to m/s
            trace.data *= 1.e-2


    return container



def get_synthetics_mtuq(data, greens, mt, Mw=None, apply_shifts=True):
    container = deepcopy(data)

    for key in ['body_waves', 'surface_waves']:
        for i in range(len(data[key])):
            synthetics = greens[key][i].get_synthetics(mt)
            for trace in container[key][i]:
                trace.weight = 1.
                component = trace.meta.channel[-1].upper()
                trace.data = synthetics.select(component=component)[0].data

                if apply_shifts:
                    if Mw==None:
                        # what is the seismic moment of the given moment tensor?
                        M0 = np.sqrt(np.sum(mt[0:3]**2.) + 0.5*np.sum(mt[3:6]**2.))
                        Mw = (np.log10(M0) - 9.1)/1.5

                    apply_magnitude_dependent_shift(trace, Mw)

    return container



def apply_magnitude_dependent_shift(trace, Mw):
    """ This type of time shift arises from the idiosyncratic way CAP 
      implements source-time function convolution. CAP's "conv" function
      results in systematic magnitude-dependent shifts between origin times
      and arrival times. This is arguably a bug. We include this function
      to allow benchmark comparisons between MTUQ synthetics (which normally
      lack such shifts) and CAP synthetics.
    """
    # the amount of the time shift is half the sum of the earthquake's rupture
    # time and rise time, as calculated by relations given in the 
    # CAP Perl wrapper
    t_offset = (cap_rupture_time(Mw) + cap_rise_time(Mw))/2.

    dt = trace.stats.delta
    nt = int(t_offset/dt)

    trace.data[nt:] = trace.data[:-nt]
    trace.data[:nt] = 0.



