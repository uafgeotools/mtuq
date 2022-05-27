
import numpy as np

import mtuq
from mtuq.util.math import radiation_coef
from mtuq.util import Null, iterable, warn
from mtuq.misfit.waveform.level2 import _to_array
from obspy.taup import TauPyModel
from mtuq.util.polarity import extract_polarity, extract_takeoff_angle

class PolarityMisfit(object):
    """ Polarity misfit function

    Compares first-motion polarities between data and synthetics

    .. rubric:: Usage

    Evaluating misfit is a two-step procedure:

    .. code::

        function_handle = Misfit(**parameters)
        values = function_handle(data, greens, sources)

    """

    def __init__(self, polarity_keyword=None, taup_model='ak135'):
        """ Function handle constructor
        Will be used to manage different kind of polarity input (np.array, pandas dataframe, data with sac header.)

        Currently only used to create the polarity misfit object.

        """

        self.polarity_keyword = polarity_keyword
        self.taup_model = taup_model

        print('Setting up default parameters')

    def __call__(self, polarity_input, greens, sources, progress_handle=Null(),
                 set_attributes=False):
        """ Evaluates polarity misfit on given data.

        """

        # Preallocate list containing source vector for faster computation.
        source_vector_list = _to_array(sources)
        # Create the array to store polarity misfit values.
        values = np.zeros((len(sources), 1))

        # Extracting measured polarities,  as a numpy array
        observed_polarities = extract_polarity(polarity_input)

        takeoff_angles = extract_takeoff_angle(greens, taup_model=self.taup_model)

        azimuths = [sta.azimuth for sta in greens]

        # Compute a NSOURCESxNSTATIONS array for quick misfit evaluation.
        predicted_polarity = np.asarray(
            [radiation_coef(source_vector_list, takeoff_angles[i], azimuths[i]) for i in range(len(observed_polarities))])

        # Misfit evaluation
        # values = (|obs|*calc) - obs
        # ((|obs|*calc) is used to filter out the un-picked station polarities)
        values = np.sum(np.abs(predicted_polarity.T *
                               np.abs(observed_polarities)-observed_polarities), axis=1)/2

        # returns a NSOURCESx1 np array
        return np.array([values]).T
