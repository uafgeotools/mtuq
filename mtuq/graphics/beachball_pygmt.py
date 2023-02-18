
#
# graphics/big_beachball.py - first motion "beachball" plots with piercing point polarities
#

# Requires pygmt intstalled

# https://github.com/obspy/obspy/issues/2388

import os
import matplotlib.pyplot as pyplot
import numpy as np

from mtuq.event import MomentTensor
from mtuq.misfit.polarity import _takeoff_angle_taup
from mtuq.util import warn

def beachball_pygmt(filename, polarity_input, greens, mt, plot_all=False, display_plot=False, taup_model = 'ak135', polarity_keyword=None):
    """ Moment tensor plot with stations polarities, implemented in PyGMT.

    .. rubric :: Input arguments


    ``filename`` (`str`):
    Name of output image file

    ``polarity_input`` (`list`, `numpy.ndarray`, `mtuq.dataset.Dataset`,
    `mtuq.greens_tensor.base.GreensTensorList, `str`):
    See mtuq.misfit.polarity.PolarityMisfit() and
    mtuq.util.polarity.extract_polarity() for a complete set of instructions. Polarity inputs should only contains [1, 0, -1] values.

    ``greens`` (mtuq.GreensTensorList): green's function list required to compute takeoff angles.

    ``mt`` (`mtuq.MomentTensor`):
    Moment tensor object

    Warning : Implemented and tested with PyGMT v0.3.1, which is still in early
    developpment. It is possible that the code might break some time in the
    future, as nerwer versions of the code are rolled-out.

    This plotting function presuppose that greens is a mtuq.GreensTensorList
    whith a valid SAC header containing the station.azimuth key. The takeoff angles is computed on the fly using obspy.taup or by reading FK Green's tensor SAC files header ('user1' key).
    """
    import pygmt
    from pygmt.helpers import build_arg_string, use_alias
    # Format the moment tensor to the expected GMT input (lon, lat, depth, mrr, mtt, mff, mrt, mrf, mtf, exponent, lon2, lat2).

    focal_mechanism = np.append(np.append([0, 0, 10], mt.as_vector()), [25, 0, 0])

    # Initialize the pygmt plot with the beachball plot
    fig = pygmt.Figure()
    fig.meca(region=[-1.2, 1.2, -1.2, 1.2], projection='m0/0/5c', scale='9.9c',
             convention="mt", G='grey50', spec=focal_mechanism, N=False, M=True)


    polarities = extract_polarity(polarity_input, polarity_keyword=polarity_keyword)
    takeoff_angles = extract_takeoff_angle(greens,taup_model=taup_model)
    azimuths = [sta.azimuth for sta in greens]

    for _i, sta in enumerate(greens):
        sta.station['takeoff_angle'] = takeoff_angles[_i]
        sta.station['azimuth'] = azimuths[_i]
        sta.station['polarity'] = polarities[_i]

    # Create a list of traces containing only the picked stations
    picked_data = [sta for i, sta in enumerate(greens) if polarities[i] != 0]
    # Create a list of traces containing the unpicked stations
    unpicked_data = [sta for i, sta in enumerate(greens) if polarities[i] == 0]

    # List the theoretical and picked radiation coefficients for comparison purposes
    predicted_polarity = np.asarray(
        [radiation_coef(mt.as_vector(), sta.station.takeoff_angle, sta.station.azimuth) for sta in picked_data])

    # Place the piercing in 4 list if they are:
    # - picked up motion matching theoretical polarity
    # - picked down motion matching theoretical polarity
    # - picked up motion not matching theoretical polarity
    # - picked down motion not matching theoretical polarity

    up_matching_data = [sta for i, sta in enumerate(
        picked_data) if sta.station.polarity == predicted_polarity[i] and sta.station.polarity == 1]
    down_matching_data = [sta for i, sta in enumerate(
        picked_data) if sta.station.polarity == predicted_polarity[i] and sta.station.polarity == -1]
    down_unmatched_data = [sta for i, sta in enumerate(
        picked_data) if sta.station.polarity != predicted_polarity[i] and sta.station.polarity == -1]
    up_unmatched_data = [sta for i, sta in enumerate(
            picked_data) if sta.station.polarity != predicted_polarity[i] and sta.station.polarity == 1]

    # Define aliases for the pygmt function. Please refer to GMT 6.2.0 `polar` function documentation for a complete overview of all the available options and option details.
    @use_alias(
        D='offset',
        J='projection',
        M='size',
        S='symbol',
        E='ext_fill',
        G='comp_fill',
        F='background',
        Qe='ext_outline',
        Qg='comp_outline',
        Qf='mt_outline',
        T='station_labels'
    )
    def _pygmt_polar(trace_list, **kwargs):
        """ Wrapper around GMT polar function. ]
        Color arguments must be in {red, green, blue, black, white} and the symbol in {a,c,d,h,i,p,s,t,x} (see GMT polar function for reference).
        """

        # Define some default values to format the plot.
        defaultKwargs = {
            'D' : '0/0',
            'J' : 'm0/0/5c',
            'M' : '9.9c',
            'T' : '+f0.18c'
            }
        kwargs = { **defaultKwargs, **kwargs }
        print(kwargs)


        colorcodes = {
            "red": "255/0/0",
            "green": "0/255/0",
            "blue": "0/0/255",
            "white":"255, 255, 255",
            "black": "0/0/0"
        }
        for key in kwargs:
            try:
                kwargs[key]=colorcodes[kwargs[key]]
            except:
                pass

        tmp_filename='polar_temp.txt'
        with open(tmp_filename, 'w') as f:
            for sta in trace_list:
                if sta.station.polarity == 1:
                    pol = '+'
                    f.write('{} {} {} {}'.format(sta.station.network+'.'+sta.station.station, sta.station.azimuth, sta.station.takeoff_angle, pol))
                    f.write('\n')

                elif sta.station.polarity == -1:
                    pol = '-'
                    f.write('{} {} {} {}'.format(sta.station.network+'.'+sta.station.station, sta.station.azimuth, sta.station.takeoff_angle, pol))
                    f.write('\n')
                else:
                    pol = '0'
                    f.write('{} {} {} {}'.format(sta.station.network+'.'+sta.station.station, sta.station.azimuth, sta.station.takeoff_angle, pol))
                    f.write('\n')
                    print('Warning !: Station ', sta.station.network+'.' +
                          sta.station.station, ' has no picked polarity')

        arg_string = " ".join([tmp_filename, build_arg_string(kwargs)])
        with pygmt.clib.Session() as session:
            session.call_module('polar',arg_string)

        os.remove(tmp_filename)


    # plotting the 4 previously lists with different symbols and colors
    _pygmt_polar(up_matching_data, symbol='t0.40c', comp_fill='green')
    _pygmt_polar(down_matching_data, symbol='i0.40c', ext_fill='blue')
    _pygmt_polar(up_unmatched_data, symbol='t0.40c', comp_fill='red')
    _pygmt_polar(down_unmatched_data, symbol='i0.40c', ext_fill='red')
    # If `plot_all` is True, will plot the unpicked stations as black crosses over the beachball plot
    if not plot_all is False:
        _pygmt_polar(unpicked_data, symbol='x0.40c', comp_outline='black')

    # fig.show(dpi=300, method="external")
    fig.savefig(filename, show=display_plot)

def extract_polarity(polarity_in, polarity_keyword=None):
    """
    Extract first motion polarity from polarity_in input.
    The function automatically detects the type of input to return an array of
    len(data) containing [-1, 0 , +1] integer values used in polarity mismatch
    misfit.

    .. rubric :: Required argument
    ``polarity_in`` (`list`, `numpy.ndarray`, `mtuq.dataset.Dataset`,
    `mtuq.greens_tensor.base.GreensTensorList, `str`):

    When parsing `list` or `numpy.ndarray`, the number of entry should match
    the number of entries in the green's tensor list.

    .. note::
    In the case of mtuq.Dataset or mtuq.GreensTensorList, the picked polarity
    info should be written in the header of the first trace of the stream for
    each station.

    """
    if isinstance(polarity_in, (mtuq.dataset.Dataset)) and polarity_keyword is not None:
        observed_polarities = np.asarray([sta[0].stats.sac[polarity_keyword] for sta in polarity_in])

    elif isinstance(polarity_in, (mtuq.greens_tensor.base.GreensTensorList)) and polarity_keyword is not None:
        observed_polarities = np.asarray([sta[0].stats.sac[polarity_keyword] for sta in polarity_in])

    elif type(polarity_in) == list or type(polarity_in) == np.ndarray:
        observed_polarities = polarity_in

    elif type(polarity_in) == str and polarity_in[-4:] == ".dat":
        observed_polarities = WeightParser(polarity_in).parse_polarity()

    else:
        raise TypeError('No polarity information was found in the polarity_input passed. Check that your polarity input is either a mtuq Dataset, a mtuq GreensTensorList, a list or numpy array or the path to the CAP weight file.')

    if not all(p == 0 or p == 1 or p==-1 for p in observed_polarities):
        raise ValueError("At least one value in polarity_input is not a 1, 0 or -1.")


    return observed_polarities


def extract_takeoff_angle(greens, taup_model='ak135'):
    """
    Extract takeoff angle from Green's function input (FK Green's function
    database), or compute it on the fly (Axisem database) using obspy.taup
    velocity model.

    """
    #Determine Green's function origin:
    solver = _get_tag(greens[0].tags, 'solver')
    if solver == 'FK':
        mode = 'FK'
    elif solver == 'AxiSEM' or solver == 'syngine':
        mode = 'taup'
    else:
        raise NotImplementedError('Greens function currently supported include AxiSEM, syngine and FK.')

    # List takeoff_angle and azimuth out of the provided data
    if mode == 'FK':
        takeoff_angles = [sta[-1].stats.sac['user1'] for sta in greens]
    elif mode == 'taup':
        model = TauPyModel(taup_model)
        takeoff_angles = [_takeoff_angle_taup(model,
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

