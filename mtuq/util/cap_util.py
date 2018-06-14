

import csv
import numpy as np
import warnings
from mtuq.util.wavelets import Wavelet


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



def parse_weight_file(filename):
    """ Parses CAP-style weight file
    """
    weights = {}
    with open(filename) as f:
        reader = csv.reader(f, delimiter=' ', skipinitialspace=True)
        for row in reader:
            id = '.'.join(row[0].split('.')[1:4])
            weights[id] = [float(w) for w in row[1:]]

    return weights


class Trapezoid(Wavelet):
    """ Trapezoid-like wavelet obtained by convolving two boxes
        Reproduces capuaf:trap.c
    """

    def __init__(self, rise_time=None):
        warnings.warn('wavelets.Trapezoid not yet tested')

        if rise_time:
            self.rise_time = rise_time
        else:
            raise ValueError


    def evaluate(self, t):
        """ Evaluates wavelet at chosen points
        """
        # rather than an anlytical formula, the following numerical procedure
        # defines the trapezoid
        if t1>t2: t1,t2 = t2,t1
        n1 = max(int(round(t1/dt)),1)
        n2 = max(int(round(t2/dt)),1)
        r = 1./(n1+n2)
        y = np.zeros(n1+n2)
        for i in range(1,n1+1):
            y[i] = y[i-1] + r
            y[-i-1] = y[i]
        for i in range(i,n2):
            y[i] = y[i-1]

        # interpolate from numerical grid to the user-supplied points
        y = np.interp(t0,y0,t)



def trapezoid_rise_time(*args, **kwargs):
    #raise NotImplementedError
    return 1.



def taper(array, taper_fraction=0.3, inplace=True):
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


def get_synthetics_cap(data, path):
    event_name = 'scak_34_20090407201255351'

    from copy import deepcopy
    from obspy import read

    bw = data['body_waves']
    sw = data['surface_waves']

    for stream in bw:
        for trace in stream:
            trace.weight = 1.
            component = trace.meta.channel[-1].upper()

            if component == 'Z':
                filename = '%s/%s.%s.BH.%d' % (path, event_name, stream.id, 7)
                trace_cap = read(filename, format='sac')[0]

            elif component == 'R':
                filename = '%s/%s.%s.BH.%d' % (path, event_name, stream.id, 9)
                trace_cap = read(filename, format='sac')[0]

            else:
                continue

            if trace.meta.npts == trace_cap.meta.npts:
                trace.data = trace_cap.data
            else:
                stream.remove(trace)
            
    for stream in sw:
        for trace in stream:
            trace.weight = 1.
            component = trace.meta.channel[-1].upper()

            if component == 'Z':
                filename = '%s/%s.%s.BH.%d' % (path, event_name, stream.id, 3)
                trace.data = read(filename, format='sac')[0].data

            if component == 'R':
                filename = '%s/%s.%s.BH.%d' % (path, event_name, stream.id, 5)
                trace.data = read(filename, format='sac')[0].data

            if component == 'T':
                filename = '%s/%s.%s.BH.%d' % (path, event_name, stream.id, 1)
                trace.data = read(filename, format='sac')[0].data

    return {
        'body_waves': bw,
        'surface_waves': sw,
        }


def get_synthetics_mtuq(greens, mt):
    synthetics_mtuq = {}
    for key in ['body_waves', 'surface_waves']:
        synthetics_mtuq[key] = greens[key].get_synthetics(mt)
    return synthetics_mtuq

