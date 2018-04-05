
import obspy
import numpy as np

from collections import defaultdict
from copy import deepcopy
from os.path import basename, exists, join
from obspy.geodetics import kilometers2degrees as km2deg
from mtuq.util.cap_util import taper, parse_weight_file
from mtuq.util.signal import cut
from mtuq.util.util import AttribDict, warn
 

class process_data(object):
    """
    CAP-style data processing function

    Processing data is a two-step procedure
        1) function_handle = process_data(filter_type=..., **filter_parameters, 
                                          pick_type=...,   **pick_parameters,
                                          window_type=..., **window_parameters,
                                          weight_type=..., **weight_parameters)

        2) processed_data = function_handle(data)

    In the first step, the user supplies a set of filtering, phase-picking,
    windowing, and weighting parameters.  In the second step, a
    single-station obspy stream is given as input and a processed stream
    returned as output.
    """

    def __init__(self,
                 filter_type=None,
                 pick_type=None,
                 window_type=None,
                 weight_type=None,
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
            self._fk_model = basename(self._fk_database)

        elif pick_type=='from_weight_file':
            assert 'weight_file' in parameters
            assert exists(parameters['weight_file'])
            self.weights = parse_weight_file(parameters['weight_file'])

        self.pick_type = pick_type
        self._picks = defaultdict(AttribDict)


        #
        # check window parameters
        #
        if window_type==None:
            warn('No window_type selected.')

        elif window_type == 'cap_bw':
            assert 'window_length' in parameters

        elif window_type == 'cap_sw':
            assert 'window_length' in parameters

        else:
             raise ValueError('Bad parameter: window_type')

        self.window_type = window_type
        self.window_length = parameters['window_length']
        self._windows = AttribDict()

        if 'time_shift_max' in parameters:
            self.time_shift_max = parameters['time_shift_max']


        #
        # check weight parameters
        #
        if weight_type==None:
            pass

        elif weight_type == 'cap_bw' or\
             weight_type == 'cap_sw':
            assert 'weight_file' in parameters
            assert exists(parameters['weight_file'])
            self.weights = parse_weight_file(parameters['weight_file'])

        else:
             raise ValueError('Bad parameter: weight_type')


        self.weight_type = weight_type



    def __call__(self, traces, overwrite=False):
        """ 
        Carries out data processing operations on obspy streams
        MTUQ GreensTensors

        input traces: all availables traces for a given station
        type traces: obspy Stream or MTUQ GreensTensor
        """
        if not hasattr(traces, 'id'):
            raise Exception("Missing station identifier")
        id = traces.id

        if not hasattr(traces, 'tag'):
            raise Exception("Missing station identifier")
        tag = traces.tag

        # station metadata are required for data processing, e.g.
        # without station location distance-depedent weighting cannont
        # be applied
        if not hasattr(traces, 'station'):
            raise Exception("Missing station metadata")
        meta = traces.station

        # overwrite existing data?
        if overwrite:
            traces = traces
        else:
            traces = deepcopy(traces)


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
        # part 2: determine phase arrival times
        #

        # Phase arrival times will be stored in a dictionary indexed by 
        # id. This allows times to be reused later when process_data is
        # called on synthetics
        if id not in self._picks:
            picks = self._picks[id]

            if self.pick_type=='from_sac_headers':
                sac_headers = meta.sac
                picks.P = sac_headers.t5
                picks.S = sac_headers.t6


            elif self.pick_type=='from_fk_database':
                sac_headers = obspy.read('%s/%s_%s/%s.grn.0' %
                    (self._fk_database,
                     self._fk_model,
                     str(int(meta.catalog_depth/1000.)),
                     str(int(meta.catalog_distance))),
                    format='sac')[0].meta.sac
                picks.P = sac_headers.t1
                picks.S = sac_headers.t2


            elif self.pick_type=='from_weight_file':
                raise NotImplementedError


        #
        # part 3a: determine window start and end times
        #

        # Start and end times will be stored in a dictionary indexed by 
        # id. This allows times to be resued later when process_data is
        # called on synthetics
        if id not in self._windows:
            origin_time = float(meta.catalog_origin_time)
            picks = self._picks[id]

            if self.window_type == 'cap_bw':
                # reproduces CAPUAF body wave window
                t1 = picks.P - 0.4*self.window_length
                t2 = t1 + self.window_length
                t1 += origin_time
                t2 += origin_time
                self._windows[id] = [t1, t2]

            elif self.window_type == 'cap_sw':
                # reproduces CAPUAF surface wave window
                t1 = picks.S - 0.3*self.window_length
                t2 = t1 + self.window_length
                t1 += origin_time
                t2 += origin_time
                self._windows[id] = [t1, t2]

            elif self.window_type == 'taup_bw':
                # determine body wave window from taup calculation
                raise NotImplementedError


        #
        # part 3b: pad greens functions relative to data
        #

        # using a longer window for greens functions than for data allows
        # time-shift corrections to be efficiently computed
        # in mtuq.misfit.cap
        if tag == 'greens_tensor':
            pad_length = self.time_shift_max

        elif tag == 'data':
            pad_length = 0.

        else:
            raise ValueError


        #
        # part 3c: cut traces along windows
        #

        window = self._windows[id]
        for trace in traces:
            cut(trace, -pad_length+window[0], window[1]+pad_length)
            taper(trace.data)


        #
        # part 4: determine weights
        #

        if self.weight_type == 'cap_bw':
            for trace in traces:
                if trace.stats.channel:
                    component = trace.stats.channel[-1].upper()

                    if id not in self.weights: 
                        trace.weight = 0.
                    elif component=='Z':
                        trace.weight = self.weights[id][1]
                    elif component=='R':
                        trace.weight = self.weights[id][2]
                    else:
                        trace.weight = 0.


        if self.weight_type == 'cap_sw':

            for trace in traces:
                if trace.stats.channel:
                    component = trace.stats.channel[-1].upper()

                    if id not in self.weights: 
                        trace.weight = 0.
                    elif component=='Z':
                        trace.weight = self.weights[id][3]
                    elif component=='R':
                        trace.weight = self.weights[id][4]
                    elif component=='T':
                        trace.weight = self.weights[id][5]
                    else:
                        trace.weight = 0.

        return traces


