
import obspy
import numpy as np
from mtuq.io.clients import fk_sac, axisem_netcdf, syngine

from matplotlib import pyplot
from os import getenv
from os.path import basename, join
from obspy import UTCDateTime
from mtuq.cap.process_data import ProcessData
from mtuq.event import Origin
from mtuq.station import Station
from mtuq.util import path_mtuq


if __name__=='__main__':
    """ Compares synthetics from three different sources
        1) AxiSEM NetCDF local database
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

    include_axisem = False
    include_fk = True
    include_syngine = True

    grid = [
       # Mrr, Mtt, Mpp, Mrt, Mrp, Mtp
       np.sqrt(1./3.)*np.array([1., 1., 1., 0., 0., 0.]), # explosion
       np.array([1., 0., 0., 0., 0., 0.]), # source 1 (on-diagonal)
       np.array([0., 1., 0., 0., 0., 0.]), # source 2 (on-diagonal)
       np.array([0., 0., 1., 0., 0., 0.]), # source 3 (on-diagonal)
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

    origin = Origin({
        'time': origin_time,
        'latitude': 61.4542007446,
        'longitude': -149.742797852,
        'depth_in_m': 33033.5998535,
        })

    stations = [
        Station({
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
        'preliminary_origin_time': origin.time,
        'preliminary_event_depth_in_m': origin.depth_in_m,
        'preliminary_event_latitude': origin.latitude,
        'preliminary_event_longitude': origin.longitude,
        }),
        Station({
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
        'preliminary_origin_time': origin.time,
        'preliminary_event_depth_in_m': origin.depth_in_m,
        'preliminary_event_latitude': origin.latitude,
        'preliminary_event_longitude': origin.longitude,
        }),
        Station({
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
        'preliminary_origin_time': origin.time,
        'preliminary_event_depth_in_m': origin.depth_in_m,
        'preliminary_event_latitude': origin.latitude,
        'preliminary_event_longitude': origin.longitude,
        }),
        Station({
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
        'preliminary_origin_time': origin.time,
        'preliminary_event_depth_in_m': origin.depth_in_m,
        'preliminary_event_latitude': origin.latitude,
        'preliminary_event_longitude': origin.longitude,
        }),
        Station({
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
        'preliminary_origin_time': origin.time,
        'preliminary_event_depth_in_m': origin.depth_in_m,
        'preliminary_event_latitude': origin.latitude,
        'preliminary_event_longitude': origin.longitude,
        }),
        Station({
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
        'preliminary_origin_time': origin.time,
        'preliminary_event_depth_in_m': origin.depth_in_m,
        'preliminary_event_latitude': origin.latitude,
        'preliminary_event_longitude': origin.longitude,
        }),
        Station({
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
        'preliminary_origin_time': origin.time,
        'preliminary_event_depth_in_m': origin.depth_in_m,
        'preliminary_event_latitude': origin.latitude,
        'preliminary_event_longitude': origin.longitude,
        }),
        Station({
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
        'preliminary_origin_time': origin.time,
        'preliminary_event_depth_in_m': origin.depth_in_m,
        'preliminary_event_latitude': origin.latitude,
        'preliminary_event_longitude': origin.longitude,
        }),
        Station({
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
        'preliminary_origin_time': origin.time,
        'preliminary_event_depth_in_m': origin.depth_in_m,
        'preliminary_event_latitude': origin.latitude,
        'preliminary_event_longitude': origin.longitude,
        }),
        Station({
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
        'preliminary_origin_time': origin.time,
        'preliminary_event_depth_in_m': origin.depth_in_m,
        'preliminary_event_latitude': origin.latitude,
        'preliminary_event_longitude': origin.longitude,
        }),
        Station({
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
        'preliminary_origin_time': origin.time,
        'preliminary_event_depth_in_m': origin.depth_in_m,
        'preliminary_event_latitude': origin.latitude,
        'preliminary_event_longitude': origin.longitude,
        }),
        Station({
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
        'preliminary_origin_time': origin.time,
        'preliminary_event_depth_in_m': origin.depth_in_m,
        'preliminary_event_latitude': origin.latitude,
        'preliminary_event_longitude': origin.longitude,
        }),
        Station({
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
        'preliminary_origin_time': origin.time,
        'preliminary_event_depth_in_m': origin.depth_in_m,
        'preliminary_event_latitude': origin.latitude,
        'preliminary_event_longitude': origin.longitude,
        }),
        Station({
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
        'preliminary_origin_time': origin.time,
        'preliminary_event_depth_in_m': origin.depth_in_m,
        'preliminary_event_latitude': origin.latitude,
        'preliminary_event_longitude': origin.longitude,
        }),
        Station({
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
        'preliminary_origin_time': origin.time,
        'preliminary_event_depth_in_m': origin.depth_in_m,
        'preliminary_event_latitude': origin.latitude,
        'preliminary_event_longitude': origin.longitude,
        }),
        Station({
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
        'preliminary_origin_time': origin.time,
        'preliminary_event_depth_in_m': origin.depth_in_m,
        'preliminary_event_latitude': origin.latitude,
        'preliminary_event_longitude': origin.longitude,
        }),
        Station({
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
        'preliminary_origin_time': origin.time,
        'preliminary_event_depth_in_m': origin.depth_in_m,
        'preliminary_event_latitude': origin.latitude,
        'preliminary_event_longitude': origin.longitude,
        }),
        Station({
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
        'preliminary_origin_time': origin.time,
        'preliminary_event_depth_in_m': origin.depth_in_m,
        'preliminary_event_latitude': origin.latitude,
        'preliminary_event_longitude': origin.longitude,
        }),
        Station({
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
        'preliminary_origin_time': origin.time,
        'preliminary_event_depth_in_m': origin.depth_in_m,
        'preliminary_event_latitude': origin.latitude,
        'preliminary_event_longitude': origin.longitude,
        }),
        Station({
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
        'preliminary_origin_time': origin.time,
        'preliminary_event_depth_in_m': origin.depth_in_m,
        'preliminary_event_latitude': origin.latitude,
        'preliminary_event_longitude': origin.longitude,
        }),
        Station({
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
        'preliminary_origin_time': origin.time,
        'preliminary_event_depth_in_m': origin.depth_in_m,
        'preliminary_event_latitude': origin.latitude,
        'preliminary_event_longitude': origin.longitude,
        }),
        Station({
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
        'preliminary_origin_time': origin.time,
        'preliminary_event_depth_in_m': origin.depth_in_m,
        'preliminary_event_latitude': origin.latitude,
        'preliminary_event_longitude': origin.longitude,
        }),
        Station({
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
        'preliminary_origin_time': origin.time,
        'preliminary_event_depth_in_m': origin.depth_in_m,
        'preliminary_event_latitude': origin.latitude,
        'preliminary_event_longitude': origin.longitude,
        }),
        Station({
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
        'preliminary_origin_time': origin.time,
        'preliminary_event_depth_in_m': origin.depth_in_m,
        'preliminary_event_latitude': origin.latitude,
        'preliminary_event_longitude': origin.longitude,
        }),
        Station({
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
        'preliminary_origin_time': origin.time,
        'preliminary_event_depth_in_m': origin.depth_in_m,
        'preliminary_event_latitude': origin.latitude,
        'preliminary_event_longitude': origin.longitude,
        }),
        Station({
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
        'preliminary_origin_time': origin.time,
        'preliminary_event_depth_in_m': origin.depth_in_m,
        'preliminary_event_latitude': origin.latitude,
        'preliminary_event_longitude': origin.longitude,
        }),]

    nstations = len(stations)
    origins = nstations*[origin]


    #
    # download Green's function databases
    #

    path_greens_fk = join(path_mtuq(), 'data/tests/benchmark_cap/greens/scak')
    # once AK135 FK database become available we'll uncomment these lines
    #if not exists(path_greens_fk):
    #    wget()
    #    unpack(path_greens_fk)

    #path_greens_axisem = ''
    #if not exists(path_greens_axisem)
    #    wget()
    #    unpack(path_greens_fk)


    #
    # specify data processing
    #

    process_bw = ProcessData(
        filter_type='Bandpass',
        freq_min= 0.1,
        freq_max= 0.333,
        pick_type='from_fk_metadata',
        fk_database=path_greens_fk,
        window_type='cap_bw',
        window_length=30.,
        )

    process_sw = ProcessData(
        filter_type='Bandpass',
        freq_min=0.025,
        freq_max=0.0625,
        pick_type='from_fk_metadata',
        fk_database=path_greens_fk,
        window_type='cap_sw',
        window_length=150.,
        )

    def process_data(greens):
        processed_greens = {}
        processed_greens['body_waves'] = greens.map(process_bw, stations, origins)
        processed_greens['surface_waves'] = greens.map(process_sw, stations, origins)
        [greens.initialize() for greens in processed_greens['body_waves']]
        [greens.initialize() for greens in processed_greens['surface_waves']]
        return processed_greens


    if include_axisem:
        print "Reading AxiSEM Greens's functions..."
        model = 'ak135'
        client_axisem = axisem_netcdf.Client(path_greens_axisem)
        greens_axisem = client_axisem.get_greens_tensors(stations, origins)
        greens_axisem = process_data(greens_axisem)


    if include_fk:
        print "Reading FK Greens's functions..."
        model = 'scak'
        client_fk = fk_sac.Client(path_greens_fk)
        greens_fk = client_fk.get_greens_tensors(stations, origins)
        greens_fk = process_data(greens_fk)


    if include_syngine:
        print "Downloading syngine Green's functions..."
        model = 'ak135'
        client_syngine = syngine.Client(model=model)
        greens_syngine = client_syngine.get_greens_tensors(stations, origins)
        greens_syngine = process_data(greens_syngine)


    print "Plotting synthetics..."
    for _i, station in enumerate(stations):
        print ' station %d of %d\n' % (_i+1, len(stations))
        # new figure object
        pyplot.figure(figsize=(30, 6.))

        for _j, mt in enumerate(grid):
            # get synthetics
            if include_axisem:
                synthetics_axisem = greens_axisem[key].get_synthetics(mt)[_i]
            if include_fk:
                synthetics_fk = greens_fk[key].get_synthetics(mt)[_i]
            if include_syngine:
                synthetics_syngine = greens_syngine[key].get_synthetics(mt)[_i]

            # get time scheme
            t = np.linspace(
                float(synthetics_syngine[0].stats.starttime),
                float(synthetics_syngine[0].stats.endtime),
                synthetics_syngine[0].stats.npts)

            for _k, component in enumerate(['Z', 'R', 'T']):
                ax = pyplot.subplot(3, len(grid), len(grid)*_k + _j + 1)

                if include_axisem:
                    stream = synthetics_axisem.select(component=component)
                    d = stream[0].data
                    pyplot.plot(t, d, color='blue', label='axisem')

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

