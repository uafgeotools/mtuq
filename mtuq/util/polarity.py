
import mtuq
import numpy as np
from obspy.taup import TauPyModel
from obspy.geodetics import kilometers2degrees as _to_deg
from mtuq.util.cap import WeightParser

def extract_polarity(polarity_in):
    """
    Extract first motion polarity from polarity_in input.
    The function automatically detects the type of input to return an array of len(data) [-1, 0 , +1] integer values used in polarity mismatch misfit.

    .. rubric :: Required argument
    ``polarity_in`` (`list`, `numpy.ndarray`, `mtuq.dataset.Dataset`,
    `mtuq.greens_tensor.base.GreensTensorList, `str`):

    When parsing `list` or `numpy.ndarray`, the number of entry should match the number of entries in the green's tensor list.

    """
    if isinstance(polarity_in, (mtuq.dataset.Dataset)) and polarity_keyword is not None:
        observed_polarities = np.asarray([sta[0].stats.sac[polarity_keyword] for sta in data])

    elif isinstance(polarity_in, (mtuq.greens_tensor.base.GreensTensorList)) and polarity_keyword is not None:
        print('detected as GreensTensorList')
        observed_polarities = np.asarray([sta[0].stats.sac[polarity_keyword] for sta in data])

    elif type(polarity_in) == list or type(polarity_in) == np.ndarray:
        print('detected as list or numpy array')
        observed_polarities = polarity_in

    elif type(polarity_in) == str and polarity_in[-4:] == ".dat":
        observed_polarities = WeightParser(polarity_in).parse_polarity()

    else:
        raise TypeError('No polarity information was found in the polarity_input passed. Check that your polarity input is either a mtuq Dataset, a mtuq GreensTensorList, a list or numpy array or the path to the CAP weight file.')
    return observed_polarities

def calculate_takeoff_angle(taup, source_depth_in_km, **kwargs):
    """
    Compute P arrival source-receiver takeoff angle, from the source-receiver geometry and a valid 1D obspy velocity model (expected in *.npz format).
    """

    try:
        arrivals = taup.get_travel_times(source_depth_in_km, **kwargs)

        phases = []
        for arrival in arrivals:
            phases += [arrival.phase.name]

        if 'p' in phases:
            return arrivals[phases.index('p')].takeoff_angle

        elif 'P' in phases:
            return arrivals[phases.index('P')].takeoff_angle
        else:
            raise Excpetion

    except:
        # if taup fails, use dummy takeoff angle
        return None

def extract_takeoff_angle(greens, taup_model='ak135'):
    """
    Extract takeoff angle from Green's function input
    """
    #Determine Green's function origin:
    solver = _get_tag(greens[0].tags, 'solver')
    if solver == 'FK':
        mode = 'FK'
    else:
        mode = 'taup'

    # List takeoff_angle and azimuth out of the provided data
    if mode == 'FK':
        takeoff_angles = [sta[-1].stats.sac['user1'] for sta in greens]
    elif mode == 'taup':
        model = TauPyModel(taup_model)
        takeoff_angles = [calculate_takeoff_angle(model,
        sta.station.sac['evdp'],
        distance_in_degree = _to_deg(sta.station.sac['dist']),
        phase_list=['p', 'P']) for sta in greens]
        # except:
        #     raise TypeError('Something went wrong with retriving takeoff_angles')
    return takeoff_angles


def _get_tag(tags, pattern):
    for tag in tags:
        parts = tag.split(':')
        if parts[0]==pattern:
            return parts[1]
    else:
        return None
