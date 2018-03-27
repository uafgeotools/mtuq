
import numpy as np

from copy import deepcopy
from os.path import exists, join
from obspy.geodetics import kilometers2degrees as km2deg
from mtuq.util.signal import cut
from mtuq.util.util import AttribDict, parse_cap_weight_file, warn


class process_data(object):
    """
    CAP-style data processing function

    Processing data is a two-step procedure
        1) function_handle = process_data(filter_type=..., **filter_parameters, 
                                          pick_type=...,   **pick_parameters,
                                          window_type=..., **window_parameters,
                                          weight_type=..., **weight_parameters)

        2) processed_stream = function_handle(data, station_metadata)

    In the first step, the user supplies a set of filtering, windowing,
    and weighting parameters.  In the second step, the user supplies a
    single-station obspy stream as input and receives a processed stream
    as output.
    """

    def __init__(self,
                 filter_type=None,
                 window_type=None,
                 weight_type=None,
                 phase_type=None,
                 **parameters):

        """ Checks data processing parameters
        """
        #
        # check filter parameters
        #
        if filter_type==None:
            warn('No filter_type selected.')

        elif filter_type == 'Bandpass':
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

        elif filter_type == 'Lowpass':
            if 'period' in parameters:
                assert 'freq' not in parameters
                parameters['freq'] = parameters['period']**-1

            if 'freq' not in parameters: raise ValueError
            assert 0 < parameters['freq']
            assert parameters['freq'] < np.inf
            self.freq = parameters['freq']


        elif filter_type == 'Highpass':
            if 'period' in parameters:
                assert 'freq' not in parameters
                parameters['freq'] = parameters['period']**-1

            if 'freq' not in parameters: raise ValueError
            assert 0 <= parameters['freq'] < np.inf
            self.freq = parameters['freq']

        else:
             raise ValueError('Bad parameter: filter_type')

        self.filter_type = filter_type


        #
        # check pick parameters
        #
        if pick_type==None:
            raise Exception

        elif pick_type=='from_sac_headers':
            pass

        elif pick_type=='from_fk_database':
            assert 'fk_database' in parameters
            self._fk_database = parameters['fk_database']

        elif pick_type=='from_weight_file':
            assert 'weight_file' in parameters
            assert exists(parameters['weight_file'])
            self.weights = parse_cap_weight_file(parameters['weight_file'])

        self.pick_type = pick_type
        self._picks = AttribDict()


        #
        # check window parameters
        #
        if window_type==None:
            warn('No window_type selected.')

        elif window_type == 'body_waves':
            assert 'window_length' in parameters

        elif window_type == 'surface_waves':
            assert 'window_length' in parameters

        else:
             raise ValueError('Bad parameter: window_type')

        self.window_type = window_type
        self.window_length = parameters['window_length']
        self._windows = AttribDict()


        #
        # check weight parameters
        #
        if weight_type==None:
            pass

        elif weight_type == 'cap_bw' or\
             weight_type == 'cap_sw':
            assert 'weight_file' in parameters
            assert exists(parameters['weight_file'])
            self.weights = parse_cap_weight_file(parameters['weight_file'])

        else:
             raise ValueError('Bad parameter: weight_type')


        self.weight_type = weight_type



    def __call__(self, traces, stats, overwrite=False):
        """ 
        Carries out data processing operations on seismic traces

        input traces: all availables traces for a given station
        type traces: obspy Stream
        input stats: station metadata
        type stats: obspy Stats
        """
        if overwrite:
            traces = traces
        else:
            traces = deepcopy(traces)

        station_id = stats.network+'.'+stats.station

        #
        # part 1: filter traces
        #
        if self.filter_type == 'Bandpass':
            for trace in traces:
                trace.detrend('demean')
                trace.detrend('linear')
                trace.taper(0.05, type='hann')
                trace.filter('bandpass', zerophase=True,
                          freqmin=self.freq_min,
                          freqmax=self.freq_max)

        elif self.filter_type == 'Lowpass':
            for trace in traces:
                trace.detrend('demean')
                trace.detrend('linear')
                trace.taper(0.05, type='hann')
                trace.filter('lowpass', zerophase=True,
                          freq=self.freq)

        elif self.filter_type == 'Highpass':
            for trace in traces:
                trace.detrend('demean')
                trace.detrend('linear')
                trace.taper(0.05, type='hann')
                trace.filter('highpass', zerophase=True,
                          freq=self.freq)

        #
        # part 2: determine phase picks
        #
        if station_id not in self._picks:
            meta = traces[0].meta
            origin_time = float(meta.starttime)

            if pick_type=='from_sac_headers':
                sac_headers = meta.sac
                self._picks.P = sac_headers.t5
                self._picks.S = sac_heasers.t6

            elif pick_type=='from_fk_database':
                self._fk_database

            elif pick_type=='from_weight_file':
                raise NotImplementedError


        #
        # part 3: determine window start and end times
        #

        # Start and end times will be stored in a dictionary indexed by 
        # station_id. This allows times determined when process_data is called
        # on data to be reused later when process_data is called on synthetics
        if station_id not in self._windows:

            if self.window_type == 'cap_bw':
                # reproduces CAPUAF body wave window
                trace = traces[0]
                sac_headers = trace.stats.sac
                origin_time = float(trace.stats.starttime)

                # CAPUAF expects the P arrival time to be in t5 header
                assert 't5' in sac_headers

                t1 = trace.stats.sac.t5 - 0.4*self.window_length
                if t1 < 0.: t1 = 0.
                t2 = t1 + self.window_length
                t1 += origin_time
                t2 += origin_time
                self._windows[station_id] = [t1, t2]

            elif self.window_type == 'cap_sw':
                # reproduces CAPUAF surface wave window
                trace = traces[0]
                sac_headers = trace.stats.sac
                origin_time = float(trace.stats.starttime)

                # CAPUAF expects the S arrival time to be in t6 header
                assert 't6' in sac_headers

                t1 = trace.stats.sac.t6 - 0.3*self.window_length
                if t1 < 0.: t1 = 0.
                t2 = t1 + self.window_length
                t1 += origin_time
                t2 += origin_time
                self._windows[station_id] = [t1, t2]

            elif self.window_type == 'taup_bw':
                # determine body wave window from taup calculation
                raise NotImplementedError


        # cut traces
        if station_id in self._windows:
            window = self._windows[station_id]
            for trace in traces:
                cut(trace, window[0], window[1])


        #
        # part 4: determine weights
        #

        if self.weight_type == 'cap_bw':

            if traces.tag == 'data':
                id = traces.id

                for trace in traces:
                    component = trace.stats.channel[-1].upper()

                    if id not in self.weights: 
                        trace.weight = 0.
                    elif component=='Z':
                        trace.weight = self.weights[id][2]
                    elif component=='R':
                        trace.weight = self.weights[id][3]
                    else:
                        trace.weight = 0.


        if self.weight_type == 'cap_sw':

            if traces.tag == 'data':
                id = traces.id

                for trace in traces:
                    component = trace.stats.channel[-1].upper()

                    if id not in self.weights: 
                        trace.weight = 0.
                    elif component=='Z':
                        trace.weight = self.weights[id][4]
                    elif component=='R':
                        trace.weight = self.weights[id][5]
                    elif component=='T':
                        trace.weight = self.weights[id][6]
                    else:
                        trace.weight = 0.

        return traces


