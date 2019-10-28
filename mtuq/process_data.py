
import csv
import obspy
import numpy as np
import warnings

from copy import deepcopy
from obspy import taup
from obspy.geodetics import gps2dist_azimuth
from os.path import basename, exists
from mtuq.util import AttribDict, warn
from mtuq.util.cap import Reader, taper
from mtuq.util.signal import cut, get_arrival, m_to_deg
 

class ProcessData(object):
    """ Performs filtering, windowing and other operations on seismic data

    .. rubric :: Usage

    Processing data is a two-step procedure:

    .. code::

        function_handle = process_data(**parameters) 
        processed_data = function_handle(data)

    In the first step, the user supplies a set of filtering, phase-picking,
    windowing, and weighting parameters.  In the second step, an obspy stream
    is given as input and a processed stream returned as output.

    See `mtuq/examples/` for a more detailed illustration of how everything
    works in practice.


    .. rubric :: Parameters

    """

    def __init__(self,
         filter_type=None,
         window_type=None,
         pick_type=None,
         window_length=None,
         padding_left=0,
         padding_right=0,
         taup_model=None,
         FK_database=None,
         FK_model=None,
         apply_statics=False,
         apply_weights=True,
         apply_scaling=True,
         scaling_power=None,
         scaling_coefficient=None,
         capuaf_file=None,
         **parameters):


        if filter_type is None:
            raise Exception("Undefined parameter: filter_type")

        if window_type is None:
            raise Exception("Undefined parameter: window_type")

        if pick_type is None:
            raise Exception("Undefined parameter: pick_type")


        self.filter_type = filter_type.lower()
        self.window_type = window_type.lower()
        self.pick_type = pick_type
        self.window_length = window_length
        self.padding_left = padding_left
        self.padding_right = padding_right
        self.taup_model = taup_model
        self.FK_database = FK_database
        self.FK_model = FK_model
        self.apply_weights = apply_weights
        self.apply_statics = apply_statics
        self.apply_scaling = apply_scaling
        self.scaling_power = scaling_power
        self.scaling_coefficient = scaling_coefficient
        self.capuaf_file = capuaf_file


        #
        # check filter parameters
        #

        if self.filter_type == 'bandpass':
            # allow filter corners to be specified in terms of either period [s]
            # or frequency [Hz]
            if 'period_min' in parameters and 'period_max' in parameters:
                assert 'freq_min' not in parameters
                assert 'freq_max' not in parameters
                parameters['freq_min'] = parameters['period_max']**-1
                parameters['freq_max'] = parameters['period_min']**-1

            if 'freq_min' not in parameters: raise ValueError
            if 'freq_max' not in parameters: raise ValueError
            assert 0 < parameters['freq_min']
            assert parameters['freq_min'] < parameters['freq_max']
            assert parameters['freq_max'] < np.inf
            self.freq_min = parameters['freq_min']
            self.freq_max = parameters['freq_max']

        elif self.filter_type == 'lowpass':
            if 'period' in parameters:
                assert 'freq' not in parameters
                parameters['freq'] = parameters['period']**-1

            if 'freq' not in parameters: raise ValueError
            assert 0 < parameters['freq']
            assert parameters['freq'] < np.inf
            self.freq = parameters['freq']

        elif self.filter_type == 'highpass':
            if 'period' in parameters:
                assert 'freq' not in parameters
                parameters['freq'] = parameters['period']**-1

            if 'freq' not in parameters: raise ValueError
            assert 0 <= parameters['freq'] < np.inf
            self.freq = parameters['freq']

        else:
             raise ValueError('Bad parameter: filter_type')


        #
        # check window parameters
        #
        #
        if self.window_type == 'body_wave':
             # nothing to check now
             pass

        elif self.window_type == 'surface_wave':
             # nothing to check now
             pass

        else:
             raise ValueError('Bad parameter: window_type')


        if self.window_length is None:
             raise ValueError('Bad parameter: window_length')

        if self.padding_left is None:
             self.padding_left = 0.

        if self.padding_right is None:
             self.padding_right = 0.

        assert self.window_length > 0

        assert self.padding_left >= 0.
        assert self.padding_right >= 0.


        #
        # check phase pick parameters
        #
        if self.pick_type == 'taup':
            assert self.taup_model is not None
            self._taup = taup.TauPyModel(self.taup_model)

        elif self.pick_type == 'FK_metadata':
            assert self.FK_database is not None
            assert exists(self.FK_database)
            if self.FK_model is None:
                self.FK_model = basename(self.FK_database)

        elif self.pick_type == 'SAC_metadata':
             pass

        elif self.pick_type == 'user_supplied':
             pass

        else:
             raise ValueError('Bad parameter: pick_type, %s' % self.pick_type)


        #
        # check weight parameters
        #
        if apply_scaling:
            if self.window_type == 'body_wave':
                if self.scaling_power is None:
                    self.scaling_power = 1.

                if self.scaling_coefficient is None:
                    self.scaling_coefficient = 1.e5


            elif self.window_type == 'surface_wave':
                if self.scaling_power is None:
                    self.scaling_power = 0.5

                if self.scaling_coefficient is None:
                    self.scaling_coefficient = 1.e5


        #
        # parse text files
        #
        if self.apply_statics or\
           self.apply_weights or\
           self.pick_type == 'user_supplied':
            assert capuaf_file is not None

        if self.capuaf_file:
            assert exists(capuaf_file)
            reader = Reader(self.capuaf_file)

        if self.apply_statics:
            self.statics = reader.parse_statics()

        if self.apply_weights:
            self.weights = reader.parse_weights()

        if self.pick_type == 'user_supplied':
            self.picks = reader.parse_picks()




    def __call__(self, traces, station=None, origin=None, overwrite=False):
        ''' 
        Carries out data processing operations on obspy streams
        MTUQ GreensTensors

        input traces: all availables traces for a given station
        type traces: obspy Stream or MTUQ GreensTensor
        '''
        if station is None:
            station = getattr(traces, 'station', None)

        if origin is None:
            origin = getattr(traces, 'origin', None)

        # overwrite existing data?
        if overwrite:
            traces = traces
        else:
            traces = deepcopy(traces)

        if not hasattr(traces, 'id'):
            raise Exception('Missing station identifier')
        id = traces.id

        # collect location information
        distance_in_m, azimuth, _ = gps2dist_azimuth(
            origin.latitude,
            origin.longitude,
            station.latitude,
            station.longitude)

        # collect time sampling information
        nt, dt = traces[0].stats.npts, traces[0].stats.delta

        # Tags can be added through dataset.add_tag to keep track of custom
        # metadata or support other customized uses. Here we use tags to
        # distinguish data from Green's functions and displacement time series 
        # from velcoity time series
        if not hasattr(traces, 'tags'):
            raise Exception('Missing tags attribute')
        tags = traces.tags


        if 'units:cm' in tags:
            # convert to meters
            for trace in traces:
                trace.data *= 1.e-2
            index = tags.index('units:cm')
            tags[index] = 'units:m'


        #
        # part 1: filter traces
        #

        if self.filter_type == 'bandpass':
            for trace in traces:
                trace.detrend('demean')
                trace.detrend('linear')
                trace.taper(0.05, type='hann')
                trace.filter('bandpass', zerophase=False,
                          freqmin=self.freq_min,
                          freqmax=self.freq_max)

        elif self.filter_type == 'lowpass':
            for trace in traces:
                trace.detrend('demean')
                trace.detrend('linear')
                trace.taper(0.05, type='hann')
                trace.filter('lowpass', zerophase=False,
                          freq=self.freq)

        elif self.filter_type == 'highpass':
            for trace in traces:
                trace.detrend('demean')
                trace.detrend('linear')
                trace.taper(0.05, type='hann')
                trace.filter('highpass', zerophase=False,
                          freq=self.freq)

        if 'type:velocity' in tags:
            # convert to displacement
            for trace in traces:
                trace.data = np.cumsum(trace.data)*dt
            index = tags.index('type:velocity')
            tags[index] = 'type:displacement'


        #
        # part 2: determine phase picks
        #

        # Phase arrival times will be stored in a dictionary indexed by 
        # id. This allows times to be reused later when process_data is
        # called on synthetics
        if self.pick_type == 'user_supplied':
            picks = self.picks[id]

        else:
            picks = dict()

            if self.pick_type=='taup':
                with warnings.catch_warnings():
                    # supress obspy warning that gets raised even when taup is 
                    # used correctly (someone should submit an obspy fix)
                    warnings.filterwarnings('ignore')
                    arrivals = self._taup.get_travel_times(
                        origin.depth_in_m/1000., 
                        m_to_deg(distance_in_m), 
                        phase_list=['p', 's', 'P', 'S'])
                try:
                    picks['P'] = get_arrival(arrivals, 'p')
                except:
                    picks['P'] = get_arrival(arrivals, 'P')
                try:
                    picks['S'] = get_arrival(arrivals, 's')
                except:
                    picks['S'] = get_arrival(arrivals, 'S')


            elif self.pick_type=='FK_metadata':
                sac_headers = obspy.read('%s/%s_%s/%s.grn.0' %
                    (self.FK_database,
                     self.FK_model,
                     str(int(np.ceil(origin.depth_in_m/1000.))),
                     str(int(np.ceil(distance_in_m/1000.)))),
                    format='sac')[0].stats.sac
                picks['P'] = sac_headers.t1
                picks['S'] = sac_headers.t2


            elif self.pick_type=='SAC_metadata':
                sac_headers = traces[0].sac
                picks['P'] = sac_headers.t5
                picks['S'] = sac_headers.t6



        #
        # part 3a: determine window start and end times
        #

        # Start and end times will be stored in a dictionary indexed by 
        # id. This allows times to be resued later when process_data is
        # called on synthetics

        if self.window_type == 'body_wave':
            # reproduces CAPUAF body wave window
            starttime = picks['P'] - 0.4*self.window_length
            endtime = starttime + self.window_length

        elif self.window_type == 'surface_wave':
            # reproduces CAPUAF surface wave window
            starttime = picks['S'] - 0.3*self.window_length
            endtime = starttime + self.window_length

        starttime += float(origin.time)
        endtime += float(origin.time)


        #
        # part 3b: apply statics
        # 
        if 'type:greens' in tags:
            for trace in traces:
                component = trace.stats.channel[-1].upper()
                key = self.window_type +'_'+ component

                try:
                    offset = self._statics[id][key]
                    starttime += offset
                    endtime += offset
                except:
                    pass


        #
        # part 3c: apply padding
        # 

        # using a longer window for Green's functions than for data allows for
        # more accurate time-shift corrections (otherwise, traces will later be
        # padded with zeros)

        if 'type:greens' in tags:
            starttime -= self.padding_left
            endtime += self.padding_right

            for trace in traces:
                setattr(trace, 'padding_left', int(self.padding_left/dt))
                setattr(trace, 'padding_right', int(self.padding_right/dt))


        #
        # part 3d: cut and taper traces
        #
        for trace in traces:
            # cuts trace and adjusts metadata
            cut(trace, starttime, endtime)
            taper(trace.data)


        #
        # part 4a: apply distance scaling
        #

        if self.apply_scaling:
            for trace in traces:
                trace.data *=\
                     (distance_in_m/self.scaling_coefficient)**self.scaling_power

        #
        # part 4b: apply user-supplied data weights
        #
        if 'type:greens' in tags:
            pass

        elif self.apply_weights:
            for trace in traces:
                try:
                    component = trace.stats.channel[-1].upper()
                    weight = self.weights[id][self.window_type+'_'+component]
                except:
                    weight = None

                if weight:
                    setattr(trace, 'weight', weight)
                else:
                    traces.remove(trace)

        return traces

