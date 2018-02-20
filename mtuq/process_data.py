
import numpy as np
import warnings

from copy import deepcopy
from os.path import join
from obspy.geodetics import kilometers2degrees as km2deg
from mtuq.util.signal import cut


class process_data(object):
    """ Processing data is a two-step procedure

            1) function_handle = process_data(filter_type=..., **filter_parameters, 
                                              window_type=..., **window_parameters,
                                              weight_type=..., **weight_parameters)
            2) processed_stream = function_handle(stream)

        In the first step, the user supplies a set of filtering, windowing,
        and weighting parameters.  In the second step, the user supplies a
        single-station obspy stream as input and receives a processed stream
        as output.

        The reason for doing it this way was to provide a clear separation
        between parameter checking and function execution, resulting in more
        readable code and clearer error messages for the user.
    """

    def __init__(self,
                 filter_type=None,
                 window_type=None,
                 weight_type=None,
                 **parameters):

        """ Checks data processing parameters
        """
        #
        # check filter parameters
        #
        if filter_type==None:
            warnings.warn('No filter_type selected.')

        elif filter_type == 'Bandpass':
            # allow corners to be specified in terms of either period [s] or
            # frequency [Hz]
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
            # allow corner to be specified in terms of either period [s] or
            # frequency [Hz]
            if 'period' in parameters:
                assert 'freq' not in parameters
                parameters['freq'] = parameters['period']**-1

            if 'freq' not in parameters: raise ValueError
            assert 0 < parameters['freq']
            assert parameters['freq'] < np.inf
            self.freq = parameters['freq']


        elif filter_type == 'Highpass':
            # allow corners to be specified in terms of either period [s] or
            # frequency [Hz]
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
        # check window parameters
        #
        if window_type==None:
            warnings.warn('No window_type selected.')

        elif window_type == 'cap_bw':
            assert 'window_length' in parameters

        elif window_type == 'cap_sw':
            assert 'window_length' in parameters

        elif window_type == 'taup_bw':
            assert 'window_length' in parameters

            assert 'taup_model' in parameters
            self.taup_model = parameters['taup_model']

            assert 'taup_phase' in parameters
            self.taup_phase = taup_phase['taup_phase']

        else:
             raise ValueError('Bad parameter: window_type')

        self.window_type = window_type
        self.window_length = parameters['window_length']
        self._windows = {}


        #
        # weighting parameters
        #
        if weight_type==None:
            pass

        elif weight_type in ['cap_bw', 'cap_sw']:
            assert 'weights_file' in parameters
            assert exists(parameters['weight_file'])

        else:
             raise ValueError('Bad parameter: weight_type')



    def __call__(self, data, overwrite=False):
        """ Carries out data processing operations on seismic traces
        """
        if overwrite:
            data = data
        else:
            data = deepcopy(data)

        #
        # filter traces
        #
        if self.filter_type == 'Bandpass':
            for trace in data:
                trace.detrend('demean')
                trace.detrend('linear')
                trace.taper(0.05, type='hann')
                trace.filter('bandpass', zerophase=True,
                          freqmin=self.freq_min,
                          freqmax=self.freq_max)

        elif self.filter_type == 'Lowpass':
            for trace in data:
                trace.detrend('demean')
                trace.detrend('linear')
                trace.taper(0.05, type='hann')
                trace.filter('lowpass', zerophase=True,
                          freq=self.freq)

        elif self.filter_type == 'Highpass':
            for trace in data:
                trace.detrend('demean')
                trace.detrend('linear')
                trace.taper(0.05, type='hann')
                trace.filter('highpass', zerophase=True,
                          freq=self.freq)

        #
        # determine window start and end times
        #

        # We previously sorted data into streams based on station, so all
        # traces in a given stream will have the same id
        stats = data[0].stats
        station_id = join(stats.network, stats.station)

        # Window start and end times will be stored in a dictionary indexed by 
        # station_id. Typically, start and end times are determined when 
        # process_data is called on data, and then these times are reused when
        # process_data is called on synthetics
        if not hasattr(self._windows, station_id):

            if self.window_type == 'cap_bw':
                # reproduces CAPUAF body wave window
                trace = data[0]
                sac_headers = trace.stats.sac
                origin_time = float(trace.stats.starttime)

                # CAPUAF expects the P arrival time to be in t5 header
                assert 't5' in sac_headers

                t1 += trace.stats.sac.t5 - 0.4*self.window_length
                t2 += trace.stats.sac.t5 + 0.6*self.window_length
                t1 += origin_time
                t2 += origin_time
                self.windows[station_id] = [t1, t2]

            elif self.window_type == 'cap_sw':
                # reproduces CAPUAF surface wave window
                trace = data[0]
                sac_headers = trace.stats.sac
                origin_time = float(trace.stats.starttime)

                # CAPUAF expects the S arrival time to be in t6 header
                assert 't6' in sac_headers

                t1 = trace.stats.sac.t6 - 0.3*self.window_length
                t2 = trace.stats.sac.t6 + 0.7*self.window_length
                t1 += origin_time
                t2 += origin_time
                self.windows[station_id] = [t1, t2]

            elif self.window_type == 'taup_bw':
                # determine body wave window from taup calculation
                raise NotImplementedError


        #
        # cut traces
        #
        if hasattr(self._windows, station_id):
            window = self._windows[station_id]
            for trace in data:
                cut(trace, window[0], window[1])


        #
        # determine weights
        #


        #
        # apply distance weights
        #
        pass

        return data


