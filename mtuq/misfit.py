
import numpy as np

def _waveform_difference_cc(dat, syn):
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


def _waveform_difference(dat, syn):
    nt = dat.stats.npts
    dt = dat.stats.delta

    dat = dat.data
    syn = syn.data
    rsd = syn[:] - dat[:]

    return np.sqrt(np.sum(rsd*rsd*dt))



def waveform_difference(dat, syn):
    # number of stations
    ns = len(syn)

    sum_misfit = 0.
    for _i in range(ns):
        for _j, component in enumerate(syn[_i]):
            sum_misfit += _waveform_difference(syn[_i][_j], dat[_i][_j])

    return sum_misfit

