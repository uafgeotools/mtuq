
import numpy as np

from mtuq.util import Null, iterable, warn


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


def radiation_coef(mt_array, takeoff_angle, azimuth):
    """ Computes P-wave radiation coefficient from a collection of sources (2D np.array with source parameter vectors), and the associated takeoff_angle and azimuth for a given station. The radiation coefficient is computed in mtuq orientation conventions.

    Based on Aki & Richards, Second edition (2009), eq. 4.88, p. 108
    """
    # Check if mt_array is a 2D array, and make it 2D if not.
    if len(mt_array.shape) == 1:
        mt_array = np.array([mt_array])

    alpha = np.deg2rad(takeoff_angle)
    az = np.deg2rad(azimuth)
    dir = np.zeros((len(mt_array),3))
    sth = np.sin(alpha)
    cth = np.cos(alpha)
    sphi = np.sin(az)
    cphi = np.cos(az)
    dir[:,0]=sth*cphi
    dir[:,1]=sth*sphi
    dir[:,2]=cth

    cth = mt_array[:,0]*dir[:,2]*dir[:,2] +\
          mt_array[:,1]*dir[:,0]*dir[:,0] +\
          mt_array[:,2]*dir[:,1]*dir[:,1] +\
       2*(mt_array[:,3]*dir[:,0]*dir[:,2] -\
          mt_array[:,4]*dir[:,1]*dir[:,2] -\
          mt_array[:,5]*dir[:,0]*dir[:,1])

    cth[cth>0] = 1
    cth[cth<0] = -1
    return cth

