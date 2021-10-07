
import numpy as np

from mtuq.util import Null, iterable, warn
from mtuq.util.math import radiation_coef


class PolarityMisfit(object):
    """ Polarity misfit function

    Compares first-motion polarities between data and synthetics

    .. rubric:: Usage

    Evaluating misfit is a two-step procedure:

    .. code::

        function_handle = Misfit(**parameters)
        values = function_handle(data, greens, sources)

    """

    def __init__(self):
        """ Function handle constructor
        Will be used to manage different kind of polarity input (np.array, pandas dataframe, data with sac header.)

        Currently only used to create the polarity misfit object.

        """

        print('Doing nothing right now')


    def __call__(self, data, greens, sources, progress_handle=Null(),
        set_attributes=False, polarity_keyword = 'user5'):
        """ Evaluates polarity misfit on given data.


        """
        # First checking that sac headers are set in odrder to run the function
        # if polarity_keyword not None:
        if all(d[0].stats.sac != None for d in data):
            polarities = [d[0].stats.sac[polarity_keyword] for d in data]
        else:
            print('SAC format expected')
            raise NotImplementedError

        # Preallocate list containing source vector for faster computation.
        source_vector_list = np.array([source.as_vector() for source in sources])

        # Create the array to store polarity misfit values.
        values = np.zeros((len(sources), 1))
        # Extracting measured polarities,  as a numpy array
        observed_polarities = np.asarray([sta[0].stats.sac[polarity_keyword] for sta in data])
        # List takeoff_angle and azimuth out of the provided data
        takeoff_angle = [sta[0].stats.sac['user3'] for sta in data]
        azimuth = [sta[0].stats.sac['az'] for sta in data]

        # Compute a NSOURCESxNSTATIONS array for quick misfit evaluation.
        predicted_polarity = np.asarray([radiation_coef(source_vector_list, takeoff_angle[i], azimuth[i]) for i in range(len(data))])

        # Misfit evaluation
        # values = (|obs|*calc) - obs
        # ((|obs|*calc) is used to filter out the un-picked station polarities)
        values = np.sum(np.abs(predicted_polarity.T*np.abs(observed_polarities)-observed_polarities), axis=1)

        # returns a NSOURCESx1 np array
        return np.array([values]).T
