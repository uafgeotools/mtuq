
import obspy
import numpy as np
from mtuq.greens_tensor import fk, instaseis, syngine

from matplotlib import pyplot
from os import getenv
from os.path import basename, join
from obspy import UTCDateTime
from obspy.core import Stats
from obspy.core.event import Origin
from mtuq.process_data.cap import ProcessData


if __name__=='__main__':
    """ Compares synthetics from three different sources
        1) Instaseis local database
        2) FK local database
        3) syngine remote database

        The comparison is not yet very meaningful because it is carried out
        using different earth models--there is not yet an earth model
        shared by all three databases
        
        Eventually we will generate FK Green's functions using a common model
        and make them available for download

        In the meantime, synthetics are not expected to match exactly,
        especially at high frequencies
    """

    # plot body waves or surface waves?
    #key = 'body_waves'
    key = 'surface_waves'

    include_instaseis = False
    include_fk = True
    include_syngine = True

    grid = [
       # Mrr, Mtt, Mpp, Mrt, Mrp, Mtp
       np.sqrt(1./3.)*np.array([1., 1., 1., 0., 0., 0.]), # explosion
       np.array([1., 0., 0., 0., 0., 0.]), # source 1 (diagonal)
       np.array([0., 1., 0., 0., 0., 0.]), # source 2 (diagonal)
       np.array([0., 0., 1., 0., 0., 0.]), # source 3 (diagonal)
       np.sqrt(1./2.)*np.array([0., 0., 0., 1., 0., 0.]), # source 4 (off-diagonal)
       np.sqrt(1./2.)*np.array([0., 0., 0., 0., 1., 0.]), # source 5 (off-diagonal)
       np.sqrt(1./2.)*np.array([0., 0., 0., 0., 0., 1.]), # source 6 (off-diagonal)
       ]
    M0 = 1.e15 # units: Newton-meter
    for mt in grid: mt *= M0


    origin_time = UTCDateTime(
        year = 2009,
        month = 4,
        day = 7,
        hour = 20,
        minute = 12,
        second = 55,
        )

    origin = Origin(
        time= origin_time,
        latitude= 61.4542007446,
        longitude= -149.742797852,
        depth= 33033.5998535,
        )

    stations = [
        Stats({
        'latitude': 61.592,
        'longitude': -149.817,
        'starttime': origin_time-100.,
        'endtime': origin_time-100.,
        'npts': 19999,
        'delta': 0.02,
        'network': 'YV',
        'station': 'BIGB',
        'location': '',
        'id': 'YV.BIGB.',
        'catalog_origin_time': origin_time,
        'catalog_depth': 33033.5998535,
        'catalog_distance': 15.8500907298,
        'catalog_azimuth': 345.527768889,
        }),
        Stats({
        'latitude': 61.245,
        'longitude': -149.540,
        'starttime': origin_time-100.,
        'endtime': origin_time-100.,
        'npts': 19999,
        'delta': 0.02,
        'network': 'YV',
        'station': 'ALPI',
        'location': '',
        'id': 'YV.ALPI.',
        'catalog_origin_time': origin_time,
        'catalog_depth': 33033.5998535,
        'catalog_distance': 25.7412111914,
        'catalog_azimuth': 154.937206124,
        }),
        Stats({
        'latitude': 61.592,
        'longitude': -149.131,
        'starttime': origin_time-100.,
        'endtime': origin_time-100.,
        'npts': 19999,
        'delta': 0.02,
        'network': 'AT',
        'station': 'PMR',
        'location': '',
        'id': 'AT.PMR.',
        'catalog_origin_time': origin_time,
        'catalog_depth': 33033.5998535,
        'catalog_distance': 36.0158231741,
        'catalog_azimuth': 64.4545662238,
        }),
        Stats({
        'latitude': 61.089,
        'longitude': -149.739,
        'starttime': origin_time-100.,
        'endtime': origin_time-100.,
        'npts': 19999,
        'delta': 0.02,
        'network': 'AK',
        'station': 'RC01',
        'location': '',
        'id': 'AK.RC01.',
        'catalog_origin_time': origin_time,
        'catalog_depth': 33033.5998535,
        'catalog_distance': 40.7071235329,
        'catalog_azimuth': 179.711452236,
        }),
        Stats({
        'latitude': 61.864,
        'longitude': -150.082,
        'starttime': origin_time-100.,
        'endtime': origin_time-100.,
        'npts': 19999,
        'delta': 0.02,
        'network': 'YV',
        'station': 'KASH',
        'location': '',
        'id': 'YV.KASH.',
        'catalog_origin_time': origin_time,
        'catalog_depth': 33033.5998535,
        'catalog_distance': 49.0297145289,
        'catalog_azimuth': 338.666195532,
        }),
        Stats({
        'latitude': 60.874,
        'longitude': -149.598,
        'starttime': origin_time-100.,
        'endtime': origin_time-100.,
        'npts': 19999,
        'delta': 0.02,
        'network': 'YV',
        'station': 'HOPE',
        'location': '',
        'id': 'YV.HOPE.',
        'catalog_origin_time': origin_time,
        'catalog_depth': 33033.5998535,
        'catalog_distance': 65.1451360155,
        'catalog_azimuth': 173.050656466,
        }),
        Stats({
        'latitude': 60.805,
        'longitude': -149.187,
        'starttime': origin_time-100.,
        'endtime': origin_time-100.,
        'npts': 19999,
        'delta': 0.02,
        'network': 'YV',
        'station': 'TUPA',
        'location': '',
        'id': 'YV.TUPA.',
        'catalog_origin_time': origin_time,
        'catalog_depth': 33033.5998535,
        'catalog_distance': 78.3428543629,
        'catalog_azimuth': 157.288772085,
        }),
        Stats({
        'latitude': 61.807,
        'longitude': -148.332,
        'starttime': origin_time-100.,
        'endtime': origin_time-100.,
        'npts': 19999,
        'delta': 0.02,
        'network': 'AK',
        'station': 'SAW',
        'location': '',
        'id': 'AK.SAW.',
        'catalog_origin_time': origin_time,
        'catalog_depth': 33033.5998535,
        'catalog_distance': 84.5340195001,
        'catalog_azimuth': 61.6653748926,
        }),
        Stats({
        'latitude': 60.672,
        'longitude': -149.481,
        'starttime': origin_time-100.,
        'endtime': origin_time-100.,
        'npts': 19999,
        'delta': 0.02,
        'network': 'YV',
        'station': 'LSUM',
        'location': '',
        'id': 'YV.LSUM.',
        'catalog_origin_time': origin_time,
        'catalog_depth': 33033.5998535,
        'catalog_distance': 88.3326378166,
        'catalog_azimuth': 170.67598173,
        }),
        Stats({
        'latitude': 60.735,
        'longitude': -150.482,
        'starttime': origin_time-100.,
        'endtime': origin_time-100.,
        'npts': 19999,
        'delta': 0.02,
        'network': 'YV',
        'station': 'MPEN',
        'location': '',
        'id': 'YV.MPEN.',
        'catalog_origin_time': origin_time,
        'catalog_depth': 33033.5998535,
        'catalog_distance': 89.5001194601,
        'catalog_azimuth': 206.794148235,
        }),
        Stats({
        'latitude': 60.551,
        'longitude': -149.594,
        'starttime': origin_time-100.,
        'endtime': origin_time-100.,
        'npts': 19999,
        'delta': 0.02,
        'network': 'YV',
        'station': 'DEVL',
        'location': '',
        'id': 'YV.DEVL.',
        'catalog_origin_time': origin_time,
        'catalog_depth': 33033.5998535,
        'catalog_distance': 100.986018723,
        'catalog_azimuth': 175.365654906,
        }),
        Stats({
        'latitude': 60.775,
        'longitude': -148.417,
        'starttime': origin_time-100.,
        'endtime': origin_time-100.,
        'npts': 19999,
        'delta': 0.02,
        'network': 'YV',
        'station': 'BLAK',
        'location': '',
        'id': 'YV.BLAK.',
        'catalog_origin_time': origin_time,
        'catalog_depth': 33033.5998535,
        'catalog_distance': 104.126076067,
        'catalog_azimuth': 136.074440283,
        }),
        Stats({
        'latitude': 60.488,
        'longitude': -150.032,
        'starttime': origin_time-100.,
        'endtime': origin_time-100.,
        'npts': 19999,
        'delta': 0.02,
        'network': 'YV',
        'station': 'RUSS',
        'location': '',
        'id': 'YV.RUSS.',
        'catalog_origin_time': origin_time,
        'catalog_depth': 33033.5998535,
        'catalog_distance': 108.781455802,
        'catalog_azimuth': 188.392980537,
        }),
        Stats({
        'latitude': 60.483,
        'longitude': -150.462,
        'starttime': origin_time-100.,
        'endtime': origin_time-100.,
        'npts': 19999,
        'delta': 0.02,
        'network': 'YV',
        'station': 'LSKI',
        'location': '',
        'id': 'YV.LSKI.',
        'catalog_origin_time': origin_time,
        'catalog_depth': 33033.5998535,
        'catalog_distance': 114.988115602,
        'catalog_azimuth': 200.106720707,
        }),
        Stats({
        'latitude': 60.662,
        'longitude': -151.277,
        'starttime': origin_time-100.,
        'endtime': origin_time-100.,
        'npts': 19999,
        'delta': 0.02,
        'network': 'YV',
        'station': 'NSKI',
        'location': '',
        'id': 'YV.NSKI.',
        'catalog_origin_time': origin_time,
        'catalog_depth': 33033.5998535,
        'catalog_distance': 121.048414873,
        'catalog_azimuth': 223.868000167,
        }),
        Stats({
        'latitude': 60.375,
        'longitude': -149.347,
        'starttime': origin_time-100.,
        'endtime': origin_time-100.,
        'npts': 19999,
        'delta': 0.02,
        'network': 'YV',
        'station': 'AVAL',
        'location': '',
        'id': 'YV.AVAL.',
        'catalog_origin_time': origin_time,
        'catalog_depth': 33033.5998535,
        'catalog_distance': 122.101412556,
        'catalog_azimuth': 169.692916508,
        }),
        Stats({
        'latitude': 60.710,
        'longitude': -147.953,
        'starttime': origin_time-100.,
        'endtime': origin_time-100.,
        'npts': 19999,
        'delta': 0.02,
        'network': 'YV',
        'station': 'PERI',
        'location': '',
        'id': 'YV.PERI.',
        'catalog_origin_time': origin_time,
        'catalog_depth': 33033.5998535,
        'catalog_distance': 127.296105568,
        'catalog_azimuth': 129.875238053,
        }),
        Stats({
        'latitude': 60.464,
        'longitude': -151.081,
        'starttime': origin_time-100.,
        'endtime': origin_time-100.,
        'npts': 19999,
        'delta': 0.02,
        'network': 'YV',
        'station': 'SOLD',
        'location': '',
        'id': 'YV.SOLD.',
        'catalog_origin_time': origin_time,
        'catalog_depth': 33033.5998535,
        'catalog_distance': 132.023528432,
        'catalog_azimuth': 213.907196416,
        }),
        Stats({
        'latitude': 61.259,
        'longitude': -152.372,
        'starttime': origin_time-100.,
        'endtime': origin_time-100.,
        'npts': 19999,
        'delta': 0.02,
        'network': 'AV',
        'station': 'SPBG',
        'location': '',
        'id': 'AV.SPBG.',
        'catalog_origin_time': origin_time,
        'catalog_depth': 33033.5998535,
        'catalog_distance': 142.331966923,
        'catalog_azimuth': 262.369826639,
        }),
        Stats({
        'latitude': 60.104,
        'longitude': -149.453,
        'starttime': origin_time-100.,
        'endtime': origin_time-100.,
        'npts': 19999,
        'delta': 0.02,
        'network': 'AK',
        'station': 'SWD',
        'location': '',
        'id': 'AK.SWD.',
        'catalog_origin_time': origin_time,
        'catalog_depth': 33033.5998535,
        'catalog_distance': 151.241318202,
        'catalog_azimuth': 173.8727135,
        }),
        Stats({
        'latitude': 60.008,
        'longitude': -149.410,
        'starttime': origin_time-100.,
        'endtime': origin_time-100.,
        'npts': 19999,
        'delta': 0.02,
        'network': 'YV',
        'station': 'HEAD',
        'location': '',
        'id': 'YV.HEAD.',
        'catalog_origin_time': origin_time,
        'catalog_depth': 33033.5998535,
        'catalog_distance': 162.120434579,
        'catalog_azimuth': 173.413686902,
        }),
        Stats({
        'latitude': 61.129,
        'longitude': -145.775,
        'starttime': origin_time-100.,
        'endtime': origin_time-100.,
        'npts': 19999,
        'delta': 0.02,
        'network': 'AK',
        'station': 'DIV',
        'location': '',
        'id': 'AK.DIV.',
        'catalog_origin_time': origin_time,
        'catalog_depth': 33033.5998535,
        'catalog_distance': 215.746816221,
        'catalog_azimuth': 97.9180072487,
        }),
        Stats({
        'latitude': 63.450,
        'longitude': -150.289,
        'starttime': origin_time-100.,
        'endtime': origin_time-100.,
        'npts': 19999,
        'delta': 0.02,
        'network': 'AK',
        'station': 'TRF',
        'location': '',
        'id': 'AK.TRF.',
        'catalog_origin_time': origin_time,
        'catalog_depth': 33033.5998535,
        'catalog_distance': 224.239111212,
        'catalog_azimuth': 353.014641671,
        }),
        Stats({
        'latitude': 60.549,
        'longitude': -145.750,
        'starttime': origin_time-100.,
        'endtime': origin_time-100.,
        'npts': 19999,
        'delta': 0.02,
        'network': 'AK',
        'station': 'EYAK',
        'location': '',
        'id': 'AK.EYAK.',
        'catalog_origin_time': origin_time,
        'catalog_depth': 33033.5998535,
        'catalog_distance': 238.379702635,
        'catalog_azimuth': 113.279988085,
        }),
        Stats({
        'latitude': 62.970,
        'longitude': -145.470,
        'starttime': origin_time-100.,
        'endtime': origin_time-100.,
        'npts': 19999,
        'delta': 0.02,
        'network': 'AK',
        'station': 'PAX',
        'location': '',
        'id': 'AK.PAX.',
        'catalog_origin_time': origin_time,
        'catalog_depth': 33033.5998535,
        'catalog_distance': 279.134053993,
        'catalog_azimuth': 50.8967398602,
        }),
        Stats({
        'latitude': 60.968,
        'longitude': -144.605,
        'starttime': origin_time-100.,
        'endtime': origin_time-100.,
        'npts': 19999,
        'delta': 0.02,
        'network': 'AK',
        'station': 'BMR',
        'location': '',
        'id': 'AK.BMR.',
        'catalog_origin_time': origin_time,
        'catalog_depth': 33033.5998535,
        'catalog_distance': 281.334781059,
        'catalog_azimuth': 98.8479894406,
        }),]


    #
    # download Green's function databases
    #

    path_greens_fk = join(getenv('CENTER1'), 'data/wf/FK_SYNTHETICS/scak')
    # once AK135 FK database become available we'll uncomment these lines
    #if not exists(path_greens_fk):
    #    wget()
    #    unpack(path_greens_fk)

    #path_greens_instaseis = ''
    #if not exists(path_greens_instaseis)
    #    wget()
    #    unpack(path_greens_fk)


    #
    # specify data processing
    #

    process_bw = ProcessData(
        filter_type='Bandpass',
        freq_min= 0.1,
        freq_max= 0.333,
        pick_type='from_fk_database',
        fk_database=path_greens_fk,
        window_type='cap_bw',
        window_length=30.,
        )

    process_sw = ProcessData(
        filter_type='Bandpass',
        freq_min=0.025,
        freq_max=0.0625,
        pick_type='from_fk_database',
        fk_database=path_greens_fk,
        window_type='cap_sw',
        window_length=150.,
        )

    def process_data(greens):
        processed_greens = {}
        processed_greens['body_waves'] = greens.map(process_bw)
        processed_greens['surface_waves'] = greens.map(process_sw)
        return processed_greens


    if include_instaseis:
        print "Reading instaseis Greens's functions..."
        model = 'ak135f_2s'
        factory_instaseis = instaseis.GreensTensorFactory(path_greens_instaseis)
        greens_instaseis = factory_instaseis(stations, origin)
        greens_instaseis = process_data(greens_instaseis)


    if include_fk:
        print "Reading FK Greens's functions..."
        model = 'scak'
        factory_fk = fk.GreensTensorFactory(path_greens_fk)
        greens_fk = factory_fk(stations, origin)
        greens_fk = process_data(greens_fk)


    if include_syngine:
        print "Downloading syngine Green's functions..."
        model = 'ak135f_2s'
        factory_syngine = syngine.GreensTensorFactory(model)
        greens_syngine = factory_syngine(stations, origin)
        greens_syngine = process_data(greens_syngine)


    print "Plotting synthetics..."
    for _i, station in enumerate(stations):
        print ' station %d of %d\n' % (_i+1, len(stations))
        # new figure object
        pyplot.figure(figsize=(30, 6.))

        for _j, mt in enumerate(grid):
            # get synthetics
            if include_instaseis:
                synthetics_instaseis = greens_instaseis[key].get_synthetics(mt)[_j]
            if include_fk:
                synthetics_fk = greens_fk[key].get_synthetics(mt)[_j]
            if include_syngine:
                synthetics_syngine = greens_syngine[key].get_synthetics(mt)[_j]

            # get time scheme
            t = np.linspace(
                float(synthetics_syngine[0].stats.starttime),
                float(synthetics_syngine[0].stats.endtime),
                synthetics_syngine[0].stats.npts)

            for _k, component in enumerate(['Z', 'R', 'T']):
                ax = pyplot.subplot(3, len(grid), len(grid)*_k + _j + 1)

                if include_instaseis:
                    stream = synthetics_instaseis.select(component=component)
                    d = stream[0].data
                    pyplot.plot(t, d, color='blue', label='instaseis')

                if include_fk:
                    stream = synthetics_fk.select(component=component)
                    d = stream[0].data
                    pyplot.plot(t, d, color='red', label='fk')

                if include_syngine:
                    stream = synthetics_syngine.select(component=component)
                    d = stream[0].data
                    pyplot.plot(t, d, color='black', label='syngine')

                # hide axis labels
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])

                # set title
                if _k==0:
                    title = 'source %d\n component %s' % (_j, component)
                else:
                    title = 'component %s' % component

                ax.set_title(title)

        filename = 'synthetics_'+station.station+'.png'
        pyplot.savefig(filename)
        pyplot.close()

