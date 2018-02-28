
import numpy as np


### functions that act on streams

class cap_bw(object):
    """ Reproduces CAP body-wave measurement
        (not finished implementing)
    """
    def __init__(self, max_shift=0.):
        self.max_shift = max_shift

    def __call__(self, dat, syn):
        ns = len(syn)
        sum_misfit = 0.
        for _i in range(ns):
            for _j, component in enumerate(syn[_i]):
                sum_misfit += waveform(syn[_i][_j], dat[_i][_j])
        return sum_misfit


class cap_sw(object):
    """ Reproduces CAP surface-wave measurement
        (not finished implementing)
    """

    def __init__(self, max_shift=0.):
        self.max_shift = max_shift


    def __call__(self, dat, syn):
        ns = len(syn)
        sum_misfit = 0.
        for _i in range(ns):
            for _j, component in enumerate(syn[_i]):
                sum_misfit += waveform(syn[_i][_j], dat[_i][_j])
        return sum_misfit



### functions that act on traces

def waveform_cc(dat, syn):
    """ Waveform difference misfit functional with time-shift correction
    """
    nt = dat.stats.npts
    dt = dat.stats.delta

    dat = dat.data
    syn = syn.data

    ioff = (np.argmax(cc)-nt+1)*dt
    if ioff <= 0:
        rsd = syn[ioff:] - dat[:-ioff]
    else:
        rsd = syn[:-ioff] - dat[ioff:]

    return np.sqrt(np.sum(rsd*rsd*dt))


def waveform(dat, syn):
    """ Waveform difference misfit functional
    """
    nt = dat.stats.npts
    dt = dat.stats.delta

    dat = dat.data
    syn = syn.data
    rsd = syn[:] - dat[:]

    return np.sqrt(np.sum(rsd*rsd*dt))


