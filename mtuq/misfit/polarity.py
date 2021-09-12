
import numpy as np

from mtuq.misfit.waveform import level0
from mtuq.util import Null, iterable, warn



class PolarityMisfit(object):
    """ Polarity misfit function

    Comparies first-motion polarities between data and synthetics

    .. rubric:: Usage

    Evaluating misfit is a two-step procedure:

    .. code::

        function_handle = Misfit(**parameters)
        values = function_handle(data, greens, sources)

    """

    def __init__(self):
        """ Function handle constructor
        """
        raise NotImplementedError


    def __call__(self, data, stations, sources, progress_handle=Null(), 
        set_attributes=False):
        """ Evaluates misfit on given data
        """

        polarities = []
        for _j, d in enumerate(data):
            polarities += [check_polarity(d)]


        values = np.zeros((len(sources), 1))

        #
        # iterate over sources
        #
        for _i, source in enumerate(sources):

            # optional progress message
            msg_handle()

            #
            # iterate over stations
            #
            for _j, d in enumerate(data):
                if polarities[_j] != simulate_polarity(stations[_j], source):
                    values[_i] += 1

        return values


def check_polarity():
    raise NotImplementedError

def simulate_polarity():
    raise NotImplementedError


