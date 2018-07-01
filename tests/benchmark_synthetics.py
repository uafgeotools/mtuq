
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
        using different earth models--there is not yet a common earth model
        shared by all three databases
        
        Eventually we will generate FK Green's functions using a shared model
        and make them available for download

        In the meantime, synthetics are not expected to match exactly
    """

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
       np.array([1., 1., 1., 0.,  0., 0.]), # explosion source
       np.array([0., 0., 0., 0., -1., 0.])  # double couple source
       ]


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

    process_data = ProcessData(
        filter_type='Bandpass',
        freq_min=0.025,
        freq_max=0.0625,
        pick_type='from_fk_database',
        fk_database=path_greens_fk,
        window_type='cap_sw',
        window_length=150.,
        padding_length=0,
        )

    #
    # the main I/O work starts now
    #

    #print "Reading instaseis Greens's functions..."
    #model = 'ak135f_5s'
    #factory_instaseis = instaseis.GreensTensorFactory(path_greens_instaseis)
    #greens_instaseis = factory_instaseis(station, origin)[0]
    #greens_instaseis = process_data(greens_fk)


    print "Reading FK Greens's functions..."
    model = 'scak'
    factory_fk = fk.GreensTensorFactory(path_greens_fk)
    greens_fk = factory_fk(station, origin)
    greens_fk = greens_fk.map(process_data)


    print "Downloading syngine Green's functions..."
    model = 'ak135f_5s'
    factory_syngine = syngine.GreensTensorFactory(model)
    greens_syngine = factory_syngine(station, origin)
    greens_syngine = greens_syngine.map(process_data)



    print "Plotting synthetics..."
    for mt in grid:
        # get synthetics
        #synthetics_instaseis = greens_instaseis.get_synthetics(mt)[0]
        synthetics_fk = greens_fk.get_synthetics(mt)[0]
        synthetics_syngine = greens_syngine.get_synthetics(mt)[0]

        # get time scheme
        t = np.linspace(
            float(synthetics_syngine[0].stats.starttime),
            float(synthetics_syngine[0].stats.endtime),
            synthetics_syngine[0].stats.npts)

        pyplot.figure(figsize=(6., 10.))
        for _i, component in enumerate(['Z', 'R', 'T']):
            #stream = synthetics_instaseis.select(component=component)
            #d0 = stream[0].data

            stream = synthetics_fk.select(component=component)
            d1 = stream[0].data

            stream = synthetics_syngine.select(component=component)
            d2 = stream[0].data

            pyplot.subplot(3, 1, _i+1)
            #pyplot.plot(t, d0, t, d1, t, d2)
            pyplot.plot(t, d1, t, d2)

        pyplot.show()


