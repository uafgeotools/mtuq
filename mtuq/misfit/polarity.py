
import numpy as np

import mtuq
from mtuq.util.math import radiation_coef
from mtuq.util import Null, iterable, warn
from mtuq.misfit.waveform.level2 import _to_array
from obspy.taup import TauPyModel
from mtuq.util.polarity import extract_polarity, extract_takeoff_angle

class PolarityMisfit(object):
    """ Polarity misfit function

    Evaluates first motion polarity by comparing observed and predicted first-motion polarities.

    .. rubric:: Usage

    Evaluating misfit is a two-step procedure:

    .. code::

        function_handle = PolarityMisfit(**parameters)
        values = function_handle(polarity_input, greens, sources)

    In the first step, the user may supply optional parameters such as the sac
    header keyword that stores the polarity inputs or the velocity model used
    to compute takeoff angles (see below for detailed argument descriptions).

    In the second step, the user supplies the polarity inputs (list, numpy
    array, mtuq Dataset or Green's function list, CAP weight files), Green's
    functions, and sources grid. This function presuppose that greens is a
    mtuq.GreensTensorList whith a valid SAC header containing the
    station.azimuth key.
    Theoretical first motion polarities are then generated and compared with
    observed first motions, and a NumPy array of misfit values is returned of
    the same length as `sources`.

    When the polarity_input is a mtuq.Dataset or a mtuq.GreensTensorList, the
    picked polarity info should be written in the header of the first trace
    of the stream of each station. The "polarity_keyword" will then have to be
    defined to point toward the right key in the SAC header of each station's
    first trace.

    .. example:

    `pmisfit = PolarityMisfit(polarity_keyword='user3')`

    will seek a +1, 0 or -1 value in
    `Stream[0].stats.sac['user3']`
    for each station stream in the mtuq.Dataset or mtuq.GreensTensorList

    .. rubric:: Parameters

    ``polarity_keyword`` (`str`): SAC header field where observed polarity is
    stored. Used when the polarity input type is mtuq.Dataset or
    mtuq.GreensTensorList (for ex: 'user3', would point to
    trace.stats.sac['user3'] entry in the SAC header).

    ``taup_model`` (`str`): Valid obspy taup model name, used to compute
    takeoff angles. If not a default obspy.taup model, taup_model should be a
    path pointing to a valid .npz velocity model file.

    .. note::

    *Convention* : Positive vertical motion should be encoded as +1 and
    negative vertical motion should be encoded as -1 and unpicked data as 0.
    (integer values [1, 0, -1])

    *Using FK* : If the Green's function were generated using FK, the takeoff
    angle should already be pre-written in the SAC files. The takeoff angle
    will thus be read from the 'user1' SAC header field of the Green's
    function, instead of being computed on the fly using obspy.taup function.

    """

    def __init__(self, polarity_keyword=None, taup_model='ak135'):
        """ Function handle constructor
        Pre-set optional arguments for custom polarity misfit inputs (sac
        header to read polarity from, velocity model to compute takeoff angles).

        """

        self.polarity_keyword = polarity_keyword
        self.taup_model = taup_model

        print('Setting up default parameters')

    def __call__(self, polarity_input, greens, sources, progress_handle=Null(),
                 set_attributes=False):
        """ Evaluates polarity misfit on given data.

        """

        # Check if the greens functions passed to polarity misfit are for Moment Tensor sources.
        if not all(green.include_mt for green in greens):
            raise NotImplementedError('Polarity misfit does not support Force sources at the moment.')

        # Preallocate list containing source vector for faster computation.
        source_vector_list = _to_array(sources)
        # Create the array to store polarity misfit values.
        values = np.zeros((len(sources), 1))

        # Extracting measured polarities,  as a numpy array
        observed_polarities = extract_polarity(polarity_input, self.polarity_keyword)

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
