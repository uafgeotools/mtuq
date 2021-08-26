
import obspy
import numpy as np
from mtuq.io.clients import FK_SAC, AxiSEM_NetCDF, SPECFEM3D_SAC, syngine

from matplotlib import pyplot
from os import getenv
from os.path import basename, join
from obspy import UTCDateTime
from mtuq.process_data import ProcessData
from mtuq.event import Origin, MomentTensor
from mtuq.station import Station
from mtuq.util import fullpath


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

    include_axisem = True
    include_fk = True
    include_specfem3d = False
    include_syngine = False

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
    grid = [MomentTensor(mt) for mt in grid]


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
        }),]


    #
    # download Green's function databases
    #

    path_greens_fk = fullpath('data/tests/benchmark_cap/greens/scak')
    # once AK135 FK database become available we'll uncomment these lines
    #if not exists(path_greens_fk):
    #    wget()
    #    unpack(path_greens_fk)

    path_greens_axisem = '/store/wf/instaseis_databases/scak_ak135f-2s'
    #if not exists(path_greens_axisem)
    #    wget()
    #    unpack(path_greens_fk)

    path_greens_specfem3d = '/home/rmodrak/data/greens/SPECFEM3D_GLOBE/Silwal2016/1D_ak135f_no_mud'

    #
    # specify data processing
    #

    process_bw = ProcessData(
        filter_type='bandpass',
        freq_min= 0.1,
        freq_max= 0.333,
        pick_type='FK_metadata',
        FK_database=path_greens_fk,
        window_type='body_wave',
        window_length=30.,
        apply_weights=False,
        )

    process_sw = ProcessData(
        filter_type='bandpass',
        freq_min=0.025,
        freq_max=0.0625,
        pick_type='FK_metadata',
        FK_database=path_greens_fk,
        window_type='surface_wave',
        window_length=150.,
        apply_weights=False,
        )

    def process_data(greens):
        processed_greens = {}
        processed_greens['body_waves'] = greens.map(process_bw)
        processed_greens['surface_waves'] = greens.map(process_sw)
        [greens._set_components(['Z','R','T']) for greens in processed_greens['body_waves']]
        [greens._set_components(['Z','R','T']) for greens in processed_greens['surface_waves']]
        return processed_greens


    if include_axisem:
        print("Reading AxiSEM Greens's functions...")
        model = 'ak135'
        client_axisem = AxiSEM_NetCDF.Client(path_greens_axisem)
        greens_axisem = client_axisem.get_greens_tensors(stations, origin)
        greens_axisem = process_data(greens_axisem)


    if include_fk:
        print("Reading FK Greens's functions...")
        model = 'scak'
        client_fk = FK_SAC.Client(path_greens_fk)
        greens_fk = client_fk.get_greens_tensors(stations, origin)
        greens_fk = process_data(greens_fk)


    if include_specfem3d:
        print("Reading SPECFEM3D/3D_GLOBE Greens's functions...")
        model = 'scak'
        client_specfem3d = SPECFEM3D_SAC.Client(path_greens_specfem3d)
        greens_specfem3d = client_specfem3d.get_greens_tensors(stations, origin)
        greens_specfem3d = process_data(greens_specfem3d)


    if include_syngine:
        print("Downloading syngine Green's functions...")
        model = 'ak135'
        client_syngine = syngine.Client(model=model)
        greens_syngine = client_syngine.get_greens_tensors(stations, origin)
        greens_syngine = process_data(greens_syngine)


    print("Plotting synthetics...")
    for _i, station in enumerate(stations):
        print(' station %d of %d\n' % (_i+1, len(stations)))
        # new figure object
        pyplot.figure(figsize=(30, 6))

        for _j, mt in enumerate(grid):
            synthetics = []

            # get synthetics
            if include_axisem:
                synthetics += [greens_axisem[key].get_synthetics(mt)[_i]]
            if include_fk:
                synthetics += [greens_fk[key].get_synthetics(mt)[_i]]
            if include_specfem3d:
                synthetics += [greens_specfem3d[key].get_synthetics(mt)[_i]]
            if include_syngine:
                synthetics += [greens_syngine[key].get_synthetics(mt)[_i]]

            # get time scheme
            t = np.linspace(
                float(synthetics[0][0].stats.starttime),
                float(synthetics[0][0].stats.endtime),
                synthetics[0][0].stats.npts)

            for _k, component in enumerate(['Z', 'R', 'T']):
                ax = pyplot.subplot(3, len(grid), len(grid)*_k + _j + 1)

                for _l in range(len(synthetics)):
                    stream = synthetics[_l].select(component=component)
                    d = stream[0].data
                    pyplot.plot(t, d)

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

