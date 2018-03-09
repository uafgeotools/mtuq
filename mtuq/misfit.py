
from collections import defaultdict
from scipy import signal
import numpy as np


### functions that act on streams

class cap_bw(object):
    """ Reproduces CAP body-wave measurement
    """
    def __init__(self, 
                 max_shift=0.,
                 weights=None):

        self.max_shift = max_shift

        if weights:
            self.weights = weights
        else:
            # weight all components equally if no weights given
            self.weights = defaultdict(lambda : 1.)


    def __call__(self, dat, syn):
        misfit = 0.
        max_shift = self.max_shift

        ni = len(dat)
        for i in range(ni):
            id = dat[i].id

            nj = len(dat[i])
            for j in range(nj):
                component = dat[i][j].stats.channel[-1]

                if component=='Z':
                    weight = self.weights[id][2]
                    if weight > 0:
                        misfit += weight *_waveform_difference_cc(
                            syn[i][j], dat[i][j], max_shift)

                if component=='R':
                    weight = self.weights[id][3]
                    if weight > 0:
                        misfit += weight *_waveform_difference_cc(
                            syn[i][j], dat[i][j], max_shift)

        return misfit


class cap_sw(object):
    """ Reproduces CAP surface-wave measurement
    """
    def __init__(self, 
                 max_shift=0., 
                 weights=None):

        self.max_shift = max_shift

        if weights:
            self.weights = weights
        else:
            # weight all components equally if no weights given
            self.weights = defaultdict(lambda : 1.)


    def __call__(self, dat, syn):
        misfit = 0.
        max_shift = self.max_shift

        ni = len(dat)
        for i in range(ni):
            id = dat[i].id

            nj = len(dat[i])
            for j in range(nj):
                component = dat[i][j].stats.channel[-1]

                if component=='Z':
                    weight = self.weights[id][4]
                    if weight > 0:
                        misfit += weight *_waveform_difference_cc(
                            syn[i][j], dat[i][j], max_shift)

                if component=='R':
                    weight = self.weights[id][5]
                    if weight > 0:
                        misfit += weight *_waveform_difference_cc(
                            syn[i][j], dat[i][j], max_shift)

                if component=='T':
                    weight = self.weights[id][6]
                    if weight > 0:
                        misfit += weight *_waveform_difference_cc(
                            syn[i][j], dat[i][j], max_shift)

        return misfit



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

    if ic == 0:
        rsd = syn - dat
    elif ic < 0:
        rsd = syn[ic:] - dat[:-ic]
    elif ic > 0:
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


