
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

    include_instaseis = False
    include_fk = True
    include_syngine = True


    origin_time = UTCDateTime(
        year = 2009,
        month = 4,
        day = 7,
        hour = 20,
        minute = 12,
        second = 55,
        )

    station = Stats({
        'longitude': -149.8174,
        'latitude': 61.5919,
        'starttime': origin_time-100.,
        'endtime': origin_time-100.,
        'npts': 19999,
        'delta': 0.02,
        'station': 'BIGB',
        'location': '',
        'id': 'YV.BIGB',
        'channels': ['BHZ', 'BHR', 'BHT'],
        'catalog_origin_time': origin_time,
        'catalog_depth': 33033.5998535,
        'catalog_distance': 15.8500907298,
        'catalog_azimuth': 345.527768889,
        })

    origin = Origin(
        time= origin_time,
        latitude= 61.4542007446,
        longitude= -149.742797852,
        depth= 33033.5998535,
        )

    grid = [
       # Mrr, Mtt, Mpp, Mrt, Mrp, Mtp
       np.array([0.816, 0.816, 0.816, 0., 0., 0.]), # explosion
       np.array([0., 0., 0., 1., 1., 0.]),          # double-couple #1
       np.array([0., 0., 0., 0., 1., 0.]),          # double-couple #2
       np.array([0., 0., 0., 0., 0., 1.]),          # double-couple #3
       ]

    M0 = 1.e15 # units: Neton-meter
    for mt in grid: mt *= M0


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
        greens_instaseis = factory_instaseis(station, origin)[0]
        greens_instaseis = process_data(greens_instaseis)


    if include_fk:
        print "Reading FK Greens's functions..."
        model = 'scak'
        factory_fk = fk.GreensTensorFactory(path_greens_fk)
        greens_fk = factory_fk(station, origin)
        greens_fk = process_data(greens_fk)


    if include_syngine:
        print "Downloading syngine Green's functions..."
        model = 'ak135f_2s'
        factory_syngine = syngine.GreensTensorFactory(model)
        greens_syngine = factory_syngine(station, origin)
        greens_syngine = process_data(greens_syngine)


    print "Plotting synthetics..."
    for _it, mt in enumerate(grid):
        print 'Moment tensor %d of %d\n' % (_it+1, len(grid))

        for key in ['body_waves', 'surface_waves']:
            # get synthetics
            if include_instaseis:
                synthetics_instaseis = greens_instaseis[key].get_synthetics(mt)[0]
            if include_fk:
                synthetics_fk = greens_fk[key].get_synthetics(mt)[0]
            if include_syngine:
                synthetics_syngine = greens_syngine[key].get_synthetics(mt)[0]

            # get time scheme
            t = np.linspace(
                float(synthetics_syngine[0].stats.starttime),
                float(synthetics_syngine[0].stats.endtime),
                synthetics_syngine[0].stats.npts)

            # new figure object
            pyplot.figure(figsize=(4, 6.))
            count = 1

            for component in ['Z', 'R', 'T']:
                ax = pyplot.subplot(3, 1, count)

                if include_instaseis:
                    stream = synthetics_instaseis.select(component=component)
                    d = stream[0].data
                    pyplot.plot(t, d, label='instaseis')

                if include_fk:
                    stream = synthetics_fk.select(component=component)
                    d = stream[0].data
                    pyplot.plot(t, d, label='fk')

                if include_syngine:
                    stream = synthetics_syngine.select(component=component)
                    d = stream[0].data
                    pyplot.plot(t, d, label='syngine')

                # hide axis labels
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])

                # set title
                title = key + ' ' + component
                ax.set_title(title)

                count += 1

            pyplot.show()

