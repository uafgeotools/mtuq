# THIS BENCHMARK USES LOCAL DATABASES THAT ONLY EXIST ON ALASKA.EDU SYSTEMS

import os
import numpy as np

from mtuq import open_db
from mtuq.graphics.waveform import plot_data_synthetics
from mtuq.process_data import ProcessData
from mtuq.event import Origin
from mtuq.station import Station
from mtuq.util import fullpath
from mtuq.util.cap import Trapezoid
from obspy import UTCDateTime
from socket import gethostname



if __name__=='__main__':
    #
    # Compares AxiSEM and FK synthetics for seven "fundamental" sources
    #

    path_greens_axisem= '/home/rmodrak/data/axisem/ak135f_mdj2-2s'
    path_greens_fk    = '/home/rmodrak/data/FK/MDJ2'
    path_weights      = '/home/rmodrak/devel/mtbench/input/AlvizuriTape2018/20130212025751273/weight_celso.dat'
    event_name=   '20130212025751273'
    model=        'MDJ2'

    process_bw = ProcessData(
        filter_type='Bandpass',
        freq_min= 0.08,
        freq_max= 0.2,
        pick_type='taup',
        taup_model='ak135',
        window_type='body_wave',
        window_length=15.,
        capuaf_file=path_weights,
        )

    process_sw = ProcessData(
        filter_type='Bandpass',
        freq_min=0.025,
        freq_max=0.0625,
        pick_type='taup',
        taup_model='ak135',
        window_type='surface_wave',
        window_length=150.,
        weight_type='surface_wave',
        capuaf_file=path_weights,
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
        year = 2006,
        month = 10, 
        day = 9,
        hour = 1,
        minute = 35,
        second = 28,
        )

    origin = [Origin({
        'time': origin_time,
        'latitude': 41.28739929199219,
        'longitude': 129.10830688476562,
        'depth_in_m': 1000.,
        })]

    #
    # We use a line a stations directly south of the event with 1 degree
    # spacing
    #

    stations = [
        Station({
        'latitude': 44.617,
        'longitude': 129.591,
        'starttime': '2006-10-09T01:30:28.010071Z',
        'endtime': '2006-10-09T01:55:27.910071Z',
        'npts': 29999,
        'delta': 0.05,
        'network': 'IC',
        'station': 'MDJ',
        'location': '00',
        'id': 'IC.MDJ.00',
        }),
        Station({
        'latitude': 36.5457,
        'longitude': 138.204,
        'starttime': '2006-10-09T01:30:28.023163Z',
        'endtime': '2006-10-09T01:55:27.923163Z',
        'npts': 29999,
        'delta': 0.05,
        'network': 'IU',
        'station': 'MAJO',
        'location': '00',
        'id': 'IC.MAJO.00',
        }),
        Station({
        'latitude': 35.35,
        'longitude': 137.029,
        'starttime': '2006-10-09T01:30:28.023224Z',
        'endtime': '2006-10-09T01:55:27.873224Z',
        'npts': 29998,
        'delta': 0.05,
        'network': 'G',
        'station': 'INU',
        'location': '00',
        'id': 'G.INU.00',
        }),
        Station({
        'latitude': 40.0183,
        'longitude': 116.168,
        'starttime': '2006-10-09T01:30:28.023163Z',
        'endtime': '2006-10-09T01:55:27.923163Z',
        'npts': 29999,
        'delta': 0.05,
        'network': 'IC',
        'station': 'BJT',
        'location': '00',
        'id': 'IC.BJT.00',
        }),]


    # figure #header

    bw_T_min = process_bw.freq_max**-1
    bw_T_max = process_bw.freq_min**-1
    sw_T_min = process_sw.freq_max**-1
    sw_T_max = process_sw.freq_min**-1

    bw_win_len = process_bw.window_length
    sw_win_len = process_sw.window_length

    #header = Header({}, shape=[4,2])
    #header[0] = 'BLACK: AxiSEM (2 s)'
    #header[1] = 'RED: FK'
    #header[2] = 'model: mdj2'
    #header[4] = 'b.w. bandpass: %.1f - %.1f s' % (bw_T_min, bw_T_max)
    #header[5] = 's.w. bandpass: %.1f - %.1f s' % (sw_T_min, sw_T_max)
    #header[6] = 'b.w. window: %.1f s' % bw_win_len
    #header[7] = 's.w. window: %.1f s' % sw_win_len


    #
    # The main work starts now
    #

    client_axisem = open_db(path_greens_axisem, format='axisem')
    greens_axisem = client_axisem.get_greens_tensors(stations, origin)

    client_fk = open_db(path_greens_fk, format='FK', model=model)
    greens_fk = client_fk.get_greens_tensors(stations, origin)

    greens_axisem.convolve(wavelet)
    greens_axisem_bw = greens_axisem.map(process_bw)
    greens_axisem_sw = greens_axisem.map(process_sw)

    greens_fk.convolve(wavelet)
    greens_fk_bw = greens_fk.map(process_bw)
    greens_fk_sw = greens_fk.map(process_sw)

    for _i, mt in enumerate(grid):
        print('%d of %d' % (_i+1, len(grid)))

        fk_bw = greens_fk_bw.get_synthetics(mt, components=['Z','R','T'])
        fk_sw = greens_fk_sw.get_synthetics(mt, components=['Z','R','T'])

        axisem_bw = greens_axisem_bw.get_synthetics(mt, components=['Z','R','T'])
        axisem_sw = greens_axisem_sw.get_synthetics(mt, components=['Z','R','T'])

        plot_data_synthetics('AxiSEM_vs_FK_'+str(_i)+'.png',
            axisem_bw, axisem_sw, fk_bw, fk_sw, stations, origin,
            station_labels=False, trace_labels=False)



