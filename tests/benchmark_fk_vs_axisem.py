# THIS BENCHMARK USES LOCAL DATABASES THAT ONLY EXIST ON CHINOOK.ALASKA.EDU

import os
import numpy as np

from os.path import join
from mtuq import open_db
from mtuq.graphics.waveform import plot_data_synthetics
from mtuq.process_data import ProcessData
from mtuq.event import Origin
from mtuq.station import Station
from mtuq.util import path_mtuq
from mtuq.util.cap import Trapezoid
from obspy import UTCDateTime
from socket import gethostname



if __name__=='__main__':
    #
    # Compares AxiSEM and FK synthetics for seven "fundamental" sources
    #

    path_greens_axisem= '/home/rmodrak/data/ak135f_scak-2s'
    path_greens_fk    = join(path_mtuq(), 'data/tests/benchmark_cap/greens/scak_all')
    path_weights      = join(path_mtuq(), 'data/tests/benchmark_cap/20090407201255351/weights.dat')
    event_name=   '20090407201255351'
    model=        'scak'

    process_bw = ProcessData(
        filter_type='Bandpass',
        freq_min= 0.08,
        freq_max= 0.2,
        pick_type='from_taup_model',
        taup_model='ak135',
        window_type='cap_bw',
        window_length=15.,
        padding_length=0,
        weight_type='cap_bw',
        cap_weight_file=path_weights,
        )

    process_sw = ProcessData(
        filter_type='Bandpass',
        freq_min=0.025,
        freq_max=0.0625,
        pick_type='from_taup_model',
        taup_model='ak135',
        window_type='cap_sw',
        window_length=150.,
        padding_length=0,
        weight_type='cap_sw',
        cap_weight_file=path_weights,
        )

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

    Mw = 4.5
    M0 = 10.**(1.5*Mw + 9.1) # units: N-m
    for mt in grid:
        mt *= np.sqrt(2)*M0

    wavelet = Trapezoid(
        magnitude=Mw)

    #
    # For the event location we use the catalog location from the
    # SilwalTape2016 main test case
    #

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

    #
    # We use a line a stations directly south of the event with 1 degree
    # spacing
    #

    stations = [
        Station({
        'latitude': 62.4542007446,
        'longitude': -149.742797852,
        'starttime': origin_time-100.,
        'endtime': origin_time-100.,
        'npts': 19999,
        'delta': 0.02,
        'network': 'AK',
        'station': 'STA1',
        'location': '',
        'id': 'AK.STA1.',
        'preliminary_origin_time': origin.time,
        'preliminary_event_depth_in_m': origin.depth_in_m,
        'preliminary_event_latitude': origin.latitude,
        'preliminary_event_longitude': origin.longitude,
        }),
        Station({
        'latitude': 63.4542007446,
        'longitude': -149.742797852,
        'starttime': origin_time-100.,
        'endtime': origin_time-100.,
        'npts': 19999,
        'delta': 0.02,
        'network': 'AK',
        'station': 'STA2',
        'location': '',
        'id': 'AK.STA2.',
        'preliminary_origin_time': origin.time,
        'preliminary_event_depth_in_m': origin.depth_in_m,
        'preliminary_event_latitude': origin.latitude,
        'preliminary_event_longitude': origin.longitude,
        }),
        Station({
        'latitude': 64.4542007446,
        'longitude': -149.742797852,
        'starttime': origin_time-100.,
        'endtime': origin_time-100.,
        'npts': 19999,
        'delta': 0.02,
        'network': 'AK',
        'station': 'STA3',
        'location': '',
        'id': 'AK.STA3.',
        'preliminary_origin_time': origin.time,
        'preliminary_event_depth_in_m': origin.depth_in_m,
        'preliminary_event_latitude': origin.latitude,
        'preliminary_event_longitude': origin.longitude,
        }),
        Station({
        'latitude': 65.4542007446,
        'longitude': -149.742797852,
        'starttime': origin_time-100.,
        'endtime': origin_time-100.,
        'npts': 19999,
        'delta': 0.02,
        'network': 'AK',
        'station': 'STA4',
        'location': '',
        'id': 'AK.STA4.',
        'preliminary_origin_time': origin.time,
        'preliminary_event_depth_in_m': origin.depth_in_m,
        'preliminary_event_latitude': origin.latitude,
        'preliminary_event_longitude': origin.longitude,
        }),
        Station({
        'latitude': 66.4542007446,
        'longitude': -149.742797852,
        'starttime': origin_time-100.,
        'endtime': origin_time-100.,
        'npts': 19999,
        'delta': 0.02,
        'network': 'AK',
        'station': 'STA5',
        'location': '',
        'id': 'AK.STA5.',
        'preliminary_origin_time': origin.time,
        'preliminary_event_depth_in_m': origin.depth_in_m,
        'preliminary_event_latitude': origin.latitude,
        'preliminary_event_longitude': origin.longitude,
        }),
        Station({
        'latitude': 67.4542007446,
        'longitude': -149.742797852,
        'starttime': origin_time-100.,
        'endtime': origin_time-100.,
        'npts': 19999,
        'delta': 0.02,
        'network': 'AK',
        'station': 'STA6',
        'location': '',
        'id': 'AK.STA6.',
        'preliminary_origin_time': origin.time,
        'preliminary_event_depth_in_m': origin.depth_in_m,
        'preliminary_event_latitude': origin.latitude,
        'preliminary_event_longitude': origin.longitude,
        }),
        Station({
        'latitude': 68.4542007446,
        'longitude': -149.742797852,
        'starttime': origin_time-100.,
        'endtime': origin_time-100.,
        'npts': 19999,
        'delta': 0.02,
        'network': 'AK',
        'station': 'STA7',
        'location': '',
        'id': 'AK.STA7.',
        'preliminary_origin_time': origin.time,
        'preliminary_event_depth_in_m': origin.depth_in_m,
        'preliminary_event_latitude': origin.latitude,
        'preliminary_event_longitude': origin.longitude,
        }),]

    nstations = len(stations)
    origins = [origin]*nstations


    # figure header

    bw_T_min = process_bw.freq_max**-1
    bw_T_max = process_bw.freq_min**-1
    sw_T_min = process_sw.freq_max**-1
    sw_T_max = process_sw.freq_min**-1

    bw_win_len = process_bw.window_length
    sw_win_len = process_sw.window_length

    header = Header({}, shape=[4,2])
    header[0] = 'BLACK: AxiSEM (2 s)'
    header[1] = 'RED: FK'
    header[2] = 'model: scak'
    header[4] = 'b.w. bandpass: %.1f - %.1f s' % (bw_T_min, bw_T_max)
    header[5] = 's.w. bandpass: %.1f - %.1f s' % (sw_T_min, sw_T_max)
    header[6] = 'b.w. window: %.1f s' % bw_win_len
    header[7] = 's.w. window: %.1f s' % sw_win_len


    #
    # The main work starts now
    #

    client_axisem = open_db(path_greens_axisem, format='axisem')
    greens_axisem = client_axisem.get_greens_tensors(stations, origins)

    client_fk = open_db(path_greens_fk, format='FK', model=model)
    greens_fk = client_fk.get_greens_tensors(stations, origins)

    greens_axisem.convolve(wavelet)
    greens_axisem_bw = greens_axisem.map(process_bw, stations, origins)
    greens_axisem_sw = greens_axisem.map(process_sw, stations, origins)

    greens_fk.convolve(wavelet)
    greens_fk_bw = greens_fk.map(process_bw, stations, origins)
    greens_fk_sw = greens_fk.map(process_sw, stations, origins)

    for _i, mt in enumerate(grid):
        print '%d of %d' % (_i+1, len(grid))

        fk_bw = greens_fk_bw.get_synthetics(mt)
        fk_sw = greens_fk_sw.get_synthetics(mt)
        axisem_bw = greens_axisem_bw.get_synthetics(mt)
        axisem_sw = greens_axisem_sw.get_synthetics(mt)

        plot_data_synthetics('AxiSEM_vs_FK_'+str(_i)+'.eps',
            axisem_bw, axisem_sw, fk_bw, fk_sw,
            mt=mt, header=header, title='%d of %d' % (_i+1, len(grid)), trace_labels=False)



