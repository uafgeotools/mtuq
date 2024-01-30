
import os
import numpy as np

from io import StringIO
from matplotlib import pyplot
from mtuq import Dataset, GreensTensorList, Origin, Station
from mtuq.event import MomentTensor
from mtuq.greens_tensor.base import GreensTensor as base
from mtuq.misfit import Misfit
from mtuq.process_data import ProcessData
from mtuq.util import fullpath, merge_dicts, save_json
from mtuq.util.cap import parse_station_codes, Trapezoid
from mtuq.wavelet import Gabor
from obspy import Stream, Trace


# for a meaningful test, observed and synthetic data must be misaligned
T_SYN = 5.
T_OBS = 10.
T_OFF = T_OBS - T_SYN


# choose any time discretization
NT = 1501
DT = 0.01
T1 = 0.

T2 = T1 + (NT-1)*DT
TL = T2 - T1
t = np.linspace(T1, T1+(NT-1)*DT, NT)
stats = {'npts': NT, 'delta': DT, 'starttime': T1, 'channel': 'Z',}


# choose any source wavelet
source_wavelet = Gabor(a=1., b=3.*np.pi)


# static time shifts are read from columns 11-12 below (for more information,
# see github.com/uafseismo/capuaf/doc/capuaf_manual)
capuaf_file = StringIO(
    "EVT.NET.STA. 10 1 1 1 1 0 0. 0. 0 0 %f 0" % (T_OFF/2))


# utility function defined relative to the chosen time discretization
def Dirac(ts):
    it = int(round((ts - T1)/DT))
    array = np.zeros(NT)
    array[it] = 1.
    return array


# just for input argument checking (values don't really matter)

origin = Origin({
    'id': 'EVT',
    'time': '1970-01-01T00:00:00.000000Z',
    'latitude': 0.,
    'longitude': 0.,
    'depth_in_m': 0.,
    })

station = Station({
    'network': 'NET',
    'station': 'STA',
    'location': '',
    'id': 'NET.STA.',
    'latitude': 0.,
    'longitude': 0.,
    })


# a single source suffices to test time-shift machinery (avoids an expensive
# grid search)
mt = MomentTensor(np.array([1., 0., 0., 0., 0., 0.])/100.)



# we override the base Green's function class to use very simple 
# Dirac delta functions, which are all we need to test the time shifts

class GreensTensor(base):

    def __init__(self):
        # The key thing is that the Dirac delta function here is offset
        # relative to the observed data below
        dirac_trace = Trace(Dirac(T_SYN), stats)
        super(GreensTensor, self).__init__([dirac_trace], station, origin)

        # Because we choose only a single source below, we only have to
        # fill in certain elements of the Green's tensor array
        self._set_components('Z')
        self._array[0,0,:] = self[0].data

    def _get_shape(self):
        return 3,6,NT

    def _precompute(self):
        pass

    def convolve(self, wavelet):
        for _i in range(3):
            self._array[0,_i,:] = np.convolve(
                self._array[0,_i,:], wavelet, 'same')


if __name__=='__main__':
    #
    # Tests time-shift corrections
    #

    #
    # MTUQ distinguishes between the following different types of 
    # time-shift corrections
    # 
    #  - "static_shift" is an initial user-supplied time shift applied during
    #    data processing
    #
    # - "cc_shift" is a subsequent cross-correlation time shift applied during 
    #   misfit evaluation
    #
    # - "time_shift" is the total correction, or in other words the sum of
    #   static and cross-correlation time shifts
    #

    #
    # For testing the above, we use the regular Green's function,
    # data processing, and misfit evaluation machinery in a very simple, 
    # nonphysical test case involving misaligned but otherwise identical
    # data and synthetics. 
    #
    # We then check whether MTUQ's static_shifts and cc_shifts produce the 
    # expected alignment between data and synthetics
    #


    # construct observed data
    trace_dirac = Trace(Dirac(T_OBS), stats)
    trace_convolved = source_wavelet.convolve(trace_dirac)
    stream = Stream(trace_convolved)

    stream.station = station
    stream.origin = origin
    data = Dataset([stream])


    # contruct Green's functions
    greens = GreensTensorList([GreensTensor()])
    greens.convolve(source_wavelet.evaluate(np.linspace(-TL/2.,+TL/2.,NT)))


    process_data = ProcessData(
        filter_type=None,
        window_type=None,
        apply_statics=True,
        apply_weights=False,
        capuaf_file=capuaf_file,
        )

    #
    # generate figure
    #
    def _plot(axis, t, d, pathspec):
        axis.plot(t, d[0][0].data, pathspec)
        axis.set_xlim(0.,15.)

    fig, axes = pyplot.subplots(4, 1)

    obs = data
    syn = greens.get_synthetics(mt)


    #
    # panel 0
    #
    axes[0].set_title('Original misaligned data (black) and synthetics (red)')

    _plot(axes[0], t, obs, '-k')
    _plot(axes[0], t, syn, '-r')


    #
    # panel 1
    #
    axes[1].set_title('Result using static time shift only')

    process_data = ProcessData(
        filter_type=None,
        window_type=None,
        apply_statics=True,
        apply_weights=False,
        capuaf_file=capuaf_file,
        )

    misfit = Misfit(
        norm='L2',
        time_shift_min=0.,
        time_shift_max=0.,
        )

    processed_data = data.map(process_data)
    processed_greens = greens.copy().map(process_data)

    attrs = misfit.collect_attributes(processed_data, processed_greens, mt)[0]['Z']
    print(attrs)

    _plot(axes[1], t, obs, '-k')
    _plot(axes[1], t + attrs.time_shift, syn, '-r')


    #
    # panel 2
    #
    axes[2].set_title('Result using cross-correlation time shift only')

    misfit = Misfit(
        norm='L2',
        time_shift_min=-abs(T_OFF),
        time_shift_max=+abs(T_OFF),
        )

    attrs = misfit.collect_attributes(data, greens.copy(), mt)[0]['Z']
    print(attrs)

    _plot(axes[2], t, obs, '-k')
    _plot(axes[2], t + attrs.time_shift, syn, '-r')


    #
    # panel 3
    #
    axes[3].set_title('Result using static and cross-correlation time shifts')

    process_data = ProcessData(
        filter_type=None,
        window_type=None,
        apply_statics=True,
        apply_weights=False,
        capuaf_file=capuaf_file,
        )

    misfit = Misfit(
        norm='L2',
        time_shift_min=-abs(T_OFF),
        time_shift_max=+abs(T_OFF),
        )

    # applies static time shift
    processed_data = data.map(process_data)
    processed_greens = greens.copy().map(process_data)

    # applies cross-correlation time shift
    attrs = misfit.collect_attributes(processed_data, processed_greens, mt)[0]['Z']
    print(attrs)

    _plot(axes[3], t, obs, '-k')
    _plot(axes[3], t + attrs.time_shift, syn, '-r')


    pyplot.savefig('tmp.png')

