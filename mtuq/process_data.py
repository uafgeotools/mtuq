
import numpy as np
import warnings
from obspy.geodetics import kilometers2degrees as km2deg

from copy import deepcopy


class process_data(object):

    def __init__(self,
                 filter_type=None,
                 window_type=None,
                 **parameters):
        """ Checks data processing parameters
        """

        #
        # check filter parameters
        #
        if filter_type == 'Bandpass':
            if 'freq_min' not in parameters: raise ValueError
            if 'freq_max' not in parameters: raise ValueError
            assert 0 < parameters['freq_min']
            assert parameters['freq_min'] < parameters['freq_max']
            assert parameters['freq_max'] < np.inf

            self.freq_min = parameters['freq_min']
            self.freq_max = parameters['freq_max']

        elif filter_type == 'Lowpass':
            if 'freq' not in parameters: raise ValueError
            assert 0 < parameters['freq_min']
            assert parameters['freq'] < np.inf

            self.freq = parameters['freq']


        elif filter_type == 'Highpass':
            if 'freq' not in parameters: raise ValueError
            assert 0 <= parameters['freq'] < np.inf

            self.freq = parameters['freq']

        elif filter_type==None:
            warnings.warn('No filter selected in process_data parameters')

        else:
             raise ValueError('Bad parameter: filter_type')

        self.filter_type = filter_type


        #
        # check window parameters
        #
        if window_type == 'cap_bw':
            raise NotImplementedError

        elif window_type == 'cap_sw':
            raise NotImplementedError

        elif window_type == 'taup_bw':
            raise NotImplementedError

            assert 'taup_model' in parameters
            self.taup_model = parameters['taup_model']

            assert 'taup_phase' in parameters
            self.taup_phase = taup_phase['taup_phase']

        elif window_type==None:
             pass

        else:
             raise ValueError('Bad parameter: filter_type')

        self.window_type = window_type


        #
        # check distance weighting parameters
        #
        pass


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
        # cut traces
        #
        if self.window_type == 'cap_bw':
            raise NotImplementedError

            # reproduce CAPUAF body wave measurement
            t1 = trace.stats.arrival_p - 0.4*self.window_length
            t2 = trace.stats.arrival_p - 0.6*self.window_length

            for trace in data:
                cut(trace, t1, t2)


        elif self.window_type == 'cap_sw':
            raise NotImplementedError

            # reproduce CAPUAF surface wave measurement
            t1 = trace.stats.arrival_p - 0.3*self.window_length
            t2 = trace.stats.arrival_p - 0.7*self.window_length

            for trace in data:
                cut(trace, t1, t2)

        elif self.window_type == 'taup_bw':
            # body wave window from taup model
            raise NotImplementedError

        #
        # apply distance weighting
        #
        pass

        return data
