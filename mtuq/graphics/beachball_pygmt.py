
#
# graphics/big_beachball.py - first motion "beachball" plots with piercing point polarities
#

# Requires pygmt intstalled

# https://github.com/obspy/obspy/issues/2388

import os
import matplotlib.pyplot as pyplot
import numpy as np

from mtuq.event import MomentTensor
from mtuq.util.math import radiation_coef
from mtuq.util import warn
from mtuq.util.polarity import extract_polarity, extract_takeoff_angle

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
