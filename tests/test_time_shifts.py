
import os
import numpy as np

from io import StringIO
from matplotlib import pyplot
from mtuq import Dataset, GreensTensorList, Origin, Station
from mtuq.event import MomentTensor
from mtuq.greens_tensor.base import GreensTensor as GreensTensorBase
from mtuq.misfit import Misfit
from mtuq.process_data import ProcessData as ProcessDataBase
from mtuq.util import AttribDict, fullpath, merge_dicts, save_json
from mtuq.util.cap import parse_station_codes, Trapezoid
from mtuq.wavelet import Gabor
from obspy import Stream, Trace
from os.path import join


# observed and synthetic data will be misaligned by the following amount
T_SYN = -2.5
T_OBS = +2.5
offset = T_OBS - T_SYN
assert offset != 0.


# use a static correction which close, but not exact
static = 0.9*offset


# time discretization
T1 = -20.
T2 = +20.
DT = 0.01

TL = T2 - T1
NT = int(round(TL/DT)) + 1
t = np.linspace(T1, T1+(NT-1)*DT, NT)
stats = AttribDict({'npts': NT, 'delta': DT, 'starttime': T1, 'channel': 'Z',})

window_length = 0.5*TL

def Dirac(ts):
    it = int(round((ts - T1)/DT))
    array = np.zeros(NT)
    array[it] = 1.
    return array


# optional source wavelet
source_wavelet = Gabor(a=1., b=2.*np.pi)


# mimics how static time shifts can be read from columns 11-12 of weight file
# (for more information, see github.com/uafseismo/capuaf/doc/capuaf_manual)
capuaf_file = StringIO("EVT.NET.STA. 10 1 1 1 1 0 0. 0. 0 0 %f 0" % static)


# dummy values
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
    'npts': stats.npts,
    'delta': stats.delta,
    'starttime': stats.starttime,
    })


# we override the base Green's function class to use only Dirac delta functions,
# which are all that we need to test time shifts
class GreensTensor(GreensTensorBase):

    def __init__(self):
        # The key thing is that the Dirac delta function here is offset
        # relative to the observed data below
        dirac_trace = Trace(Dirac(T_SYN), stats)
        super(GreensTensor, self).__init__([dirac_trace], station, origin)
        self.tags += ['type:greens']
        self.tags += ['units:m']

    def _precompute(self):
        # Because we generate synthetics from only a single source (see mt
        # definition below), we only have to fill in certain elements
        self._array[0,0,:] = self[0].data


# a single source suffices to test time shifts
mt = MomentTensor(np.array([1., 0., 0., 0., 0., 0.]))


# we simplify how the data processing class is initialized, as we are using it
# only to apply static time shifts
class ProcessData(ProcessDataBase):

    def __init__(self, apply_statics=False):

        super(ProcessData, self).__init__(
            apply_statics=apply_statics,
            window_length=window_length,
            window_type='min_max',
            v_min=1.,v_max=1.,
            apply_scaling=False,
            apply_weights=False,
            capuaf_file=capuaf_file)

    def __call__(self, traces):
        return super(ProcessData, self).__call__(
           traces, station=station, origin=origin)


def add_panel(axis, data, greens, apply_statics=False, apply_cc=False):

    process_data = ProcessData(apply_statics)

    # static time shifts applied now
    data = data.copy().map(process_data)
    greens = greens.copy().map(process_data)

    if apply_cc:
        misfit = Misfit(time_shift_min=-abs(offset), time_shift_max=+abs(offset))
    else:
        misfit = Misfit(time_shift_min=0., time_shift_max=0.)

    # cross-correlation time shifts applied now
    attrs = _get_attrs(data, greens, misfit)

    # time discretization
    t1, t2, nt = get_time_sampling(data)
    t = np.linspace(t1, t2, nt)

    _plot_dat(axis, t, data, attrs)
    _plot_syn(axis, t, _get_synthetics(greens, mt), attrs)

    _annotate(axis, attrs)

    axis.set_xlabel('Time (s)')

    axis.set_xlim(-window_length/2., +window_length/2.)
    axis.get_yaxis().set_ticks([])


def collect_attributes(data, greens, apply_statics=False, apply_cc=False):

    process_data = ProcessData(apply_statics)
    
    data = data.copy().map(process_data)
    greens = greens.copy().map(process_data)
    
    if apply_cc:
        misfit = Misfit(time_shift_min=-abs(offset), time_shift_max=+abs(offset))
    else:
        misfit = Misfit(time_shift_min=0., time_shift_max=0.)
    
    return _get_attrs(data, greens, misfit)



def _plot_dat(axis, t, data, attrs, pathspec='-k'):
    stream = data[0]
    trace = data[0][0]
    stats = trace.stats

    axis.plot(t, trace.data, pathspec)


def _plot_syn(axis, t, data, attrs, pathspec='-r'):
    stream = data[0]
    trace = data[0][0]
    stats = trace.stats

    idx1 = attrs.idx_start
    idx2 = attrs.idx_stop
    axis.plot(t, trace.data[idx1:idx2], pathspec)


def _annotate(axis, attrs):

    text = 'static_shift: %.1f' % attrs.static_shift
    pyplot.text(0.02, 0.85, text, transform=axis.transAxes)

    text = 'time_shift: %.1f' % attrs.time_shift
    pyplot.text(0.02, 0.70, text, transform=axis.transAxes)

    text = 'total_shift: %.1f' % attrs.total_shift
    pyplot.text(0.02, 0.55, text, transform=axis.transAxes)


def _get_synthetics(greens, mt):
    return greens.get_synthetics(mt, components=['Z'])


def _get_attrs(data, greens, misfit):
    return misfit.collect_attributes(data, greens, mt)[0]['Z']


def get_time_sampling(data):
    stream = data[0]
    t1 = float(stream[0].stats.starttime)
    t2 = float(stream[0].stats.endtime)
    nt = stream[0].data.size
    return t1, t2, nt



if __name__=='__main__':
    #
    # Tests time-shift corrections
    #

    #
    # MTUQ distinguishes between the following different types of 
    # time-shift corrections
    # 
    # - "static_shift" is an initial user-supplied time shift applied during
    #   data processing
    #
    # - "time_shift" is a subsequent cross-correlation time shift applied during 
    #   misfit evaluation
    #
    # - "total_shift" is the total correction, or in other words the sum of
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

    # by default, the script runs with figure generation and error checking
    # turned on
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_checks', action='store_true')
    parser.add_argument('--no_figures', action='store_true')
    args = parser.parse_args()
    run_checks = (not args.no_checks)
    run_figures = (not args.no_figures)


    # construct observed data
    trace_dirac = Trace(Dirac(T_OBS), stats)
    trace_convolved = source_wavelet.convolve(trace_dirac)

    trace = trace_convolved

    stream = Stream(trace_convolved)
    stream.station = station
    stream.origin = origin

    data = Dataset([stream])


    # construct observed data
    greens = GreensTensorList([GreensTensor()])
    greens.convolve(source_wavelet)


    if run_checks:

        attrs = collect_attributes(data, greens, apply_statics=True, apply_cc=False)
        assert attrs.total_shift == attrs.static_shift == static

        attrs = collect_attributes(data, greens, apply_statics=False, apply_cc=True)
        assert attrs.total_shift == attrs.time_shift == offset

        attrs = collect_attributes(data, greens, apply_statics=True, apply_cc=True)
        assert attrs.static_shift == static
        assert attrs.time_shift == offset - static
        assert attrs.total_shift == offset


    if run_figures:
        # loads fonts
        import mtuq.graphics

        fig, axes = pyplot.subplots(4, 1, figsize=(8.,10.))
        pyplot.subplots_adjust(hspace=0.75)


        print('')
        print('Panel 0')
        print('')

        axes[0].set_title('Original misaligned synthetics (red) and data (black)')

        add_panel(axes[0], data, greens, apply_statics=False, apply_cc=False)


        print('')
        print('Panel 1')
        print('')

        axes[1].set_title('Result using static time shift only')

        add_panel(axes[1], data, greens, apply_statics=True, apply_cc=False)


        print('')
        print('Panel 2')
        print('')

        axes[2].set_title('Result using cross-correlation time shift only')

        add_panel(axes[2], data, greens, apply_statics=False, apply_cc=True)


        print('')
        print('Panel 3')
        print('')

        axes[3].set_title('Result using static and cross-correlation time shifts')

        add_panel(axes[3], data, greens, apply_statics=True, apply_cc=True)


        pyplot.savefig('time_shifts.png')


