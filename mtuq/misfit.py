
from scipy import signal
import numpy as np


### functions that act on streams

class cap_bw(object):
    """ Reproduces CAP body-wave measurement
        (not finished implementing)
    """
    def __init__(self, max_shift=0.):
        self.max_shift = max_shift


    def __call__(self, data, synthetics):
        ns = len(synthetics)

        sum_misfit = 0.
        for _i in range(ns):
            syn, dat = data[_i], synthetics[_i]
            for _j, component in enumerate(syn):
                sum_misfit += _waveform_difference_cc(syn[_j], dat[_j], ts)
        return sum_misfit


class cap_sw(object):
    """ Reproduces CAP surface-wave measurement
        (not finished implementing)
    """
    def __init__(self, max_shift=0.):
        self.max_shift = max_shift


    def __call__(self, data, synthetics):
        ns = len(synthetics)
        tmax = self.max_shift

        sum_misfit = 0.
        for _i in range(ns):
            syn, dat = data[_i], synthetics[_i]
            for _j, component in enumerate(syn):
                sum_misfit += _waveform_difference_cc(syn[_j], dat[_j], tmax)
        return sum_misfit



### functions that act on traces

def _waveform_difference_cc(syn, dat, max_shift, mode=1):
    """ Waveform difference misfit functional with time-shift correction
    """
    nt = dat.stats.npts
    dt = dat.stats.delta
    nc = int(2.*max_shift/dt)

    dat = dat.data
    syn = syn.data

    if mode==1:
        # frequency-domain implementation
        cc = signal.fftconvolve(dat[:nc], syn[nc::-1], mode='same')
        ic = (np.argmax(cc)-nt+1)

    elif mode==2:
        # time-domain implementation
        cc = np.correlate(dat[:nc], syn[:nc], 'same')
        ic = (np.argmax(cc)-nt+1)

    if ic <= 0:
        rsd = syn[ic:] - dat[:-ic]
    else:
        rsd = syn[:-ic] - dat[ic:]

    return np.sqrt(np.sum(rsd*rsd*dt))


def _waveform_difference(syn, dat):
    """ Waveform difference misfit functional
    """
    nt = dat.stats.npts
    dt = dat.stats.delta

    dat = dat.data
    syn = syn.data

    rsd = syn[:] - dat[:]

    return np.sqrt(np.sum(rsd*rsd*dt))


