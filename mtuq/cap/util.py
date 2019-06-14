

import csv
import numpy as np
import obspy
from copy import deepcopy
from mtuq.graphics.waveform import Header
from mtuq.wavelet import EarthquakeTrapezoid



#
# the following functions allow reading in and performing other operations with
# CAP-style weight files. Such files can be used to control the weight on  
# individual stations and components in a moment tensor inversion
#

def parse_weight_file(filename):
    """ Parses CAP-style weight file
    """
    weights = {}
    with open(filename) as file:
        reader = csv.reader(file, delimiter=' ', skipinitialspace=True)
        for row in reader:
            id = '.'.join(row[0].split('.')[1:4])
            weights[id] = [float(w) for w in row[1:]]

    return weights


def remove_unused_stations(dataset, filename):
    """ Removes any stations not listed in CAP weight file or any stations
        with all zero weights
    """
    weights = parse_weight_file(filename)

    unused = []
    for stream in dataset:
        id = stream.id
        if id not in weights:
             unused+=[id]
             continue

        if weights[id][1]==weights[id][2]==\
           weights[id][3]==weights[id][4]==weights[id][5]==0.:
             unused+=[id]

    for id in unused:
        dataset.remove(id)


def quick_header(event_name, process_bw, process_sw, misfit_bw, misfit_sw,
    model, solver, mt, depth_in_m):

    M0 = np.sqrt(0.5*np.sum(mt[0:3]**2.) + np.sum(mt[3:6]**2.))
    Mw = (np.log10(M0) - 9.1)/1.5

    order = misfit_bw.order
    assert order==misfit_sw.order
    norm = '$L%s$' % order

    bw_T_min = process_bw.freq_max**-1
    bw_T_max = process_bw.freq_min**-1
    sw_T_min = process_sw.freq_max**-1
    sw_T_max = process_sw.freq_min**-1

    bw_win_len = process_bw.window_length
    sw_win_len = process_sw.window_length

    return Header(
        shape=np.array([4,4]), 
        items={
            0: '$M_w$: %3.2f' % Mw,
            1: 'depth: %.1f km' % (depth_in_m/1000.),
            #2: 'CLVD: %.0f' % 0.,
            #3: 'ISO: %.0f' % 0.,
            4: 'model: %s' % model,
            5: 'solver: %s' % solver,
            #6: 'norm: %s' % norm,
            8: 'b.w. bandpass: %.1f - %.1f s' % (bw_T_min, bw_T_max),
            10: 's.w. bandpass: %.1f - %.1f s' % (sw_T_min, sw_T_max),
            12: 'b.w. window: %.1f s' % bw_win_len,
            14: 's.w. window: %.1f s' % sw_win_len})


#
# These rupture and rise time utilties can be used to generate source-time
# functions which match those defined in the cap Perl wrapper.  For use in
# conjunction with mtuq/util/wavelets.py:Trapezoid
#

def Trapezoid(magnitude=None):
    if magnitude==None:
        raise ValueError

    return EarthquakeTrapezoid(
        rupture_time=cap_rupture_time(magnitude),
        rise_time=cap_rise_time(magnitude))


def cap_rupture_time(Mw):
    rupture_time = np.floor(
        10.**(0.5*Mw - 2.5) + 0.5)

    if rupture_time < 1.:
        rupture_time = 1.

    if rupture_time > 9.:
        rupture_time = 9.

    return rupture_time


def cap_rise_time(Mw):
    return 0.5*cap_rupture_time(Mw)



#
# the following functions help reproduce cap signal processing functionality.
# See also mtuq/process_data/cap.py
#

def taper(array, taper_fraction=0.3, inplace=True):
    """ Reproduces CAP taper behavior. Similar to obspy Tukey?
    """
    if inplace:
        array = array
    else:
        array = np.copy(array)
    f = taper_fraction
    M = int(round(f*len(array)))
    I = np.linspace(0.,1.,M)
    taper = 0.5*(1-np.cos(np.pi*I))
    array[:M] *= taper
    array[-1:-M-1:-1] *= taper
    if not inplace:
        return array


#
# the following functions are used in benchmark comparisons between CAP and 
# MTUQ synthetics. See tests/benchmark_cap_mtuq.py
#

def get_synthetics_cap(dummy_bw, dummy_sw, path, event_name):
    synthetics_bw = deepcopy(dummy_bw)
    synthetics_sw = deepcopy(dummy_sw)

    for stream in synthetics_bw:
        for trace in stream:
            trace.weight = 1.
            component = trace.meta.channel[-1].upper()

            if component == 'Z':
                filename = '%s/%s.%s.BH.%d' % (path, event_name, stream.id, 9)
                trace_cap = obspy.read(filename, format='sac')[0]

            elif component == 'R':
                filename = '%s/%s.%s.BH.%d' % (path, event_name, stream.id, 7)
                trace_cap = obspy.read(filename, format='sac')[0]

            else:
                continue

            if trace.meta.npts == trace_cap.meta.npts:
                trace.data = trace_cap.data
                # convert from cm/s to m/s
                trace.data *= 1.e-2

            else:
                stream.remove(trace)
            
    for stream in synthetics_sw:
        for trace in stream:
            trace.weight = 1.
            component = trace.meta.channel[-1].upper()

            if component == 'Z':
                filename = '%s/%s.%s.BH.%d' % (path, event_name, stream.id, 5)
                trace.data = obspy.read(filename, format='sac')[0].data

            if component == 'R':
                filename = '%s/%s.%s.BH.%d' % (path, event_name, stream.id, 3)
                trace.data = obspy.read(filename, format='sac')[0].data

            if component == 'T':
                filename = '%s/%s.%s.BH.%d' % (path, event_name, stream.id, 1)
                trace.data = obspy.read(filename, format='sac')[0].data

            # convert from cm/s to m/s
            trace.data *= 1.e-2

    return synthetics_bw, synthetics_sw


def get_data_cap(dummy_bw, dummy_sw, path, event_name):
    data_bw = deepcopy(dummy_bw)
    data_sw = deepcopy(dummy_sw)

    for stream in data_bw:
        for trace in stream:
            trace.weight = 1.
            component = trace.meta.channel[-1].upper()

            if component == 'Z':
                filename = '%s/%s.%s.BH.%d' % (path, event_name, stream.id, 8)
                trace_cap = obspy.read(filename, format='sac')[0]

            elif component == 'R':
                filename = '%s/%s.%s.BH.%d' % (path, event_name, stream.id, 6)
                trace_cap = obspy.read(filename, format='sac')[0]

            else:
                continue

            if trace.meta.npts == trace_cap.meta.npts:
                trace.data = trace_cap.data
                # convert from cm/s to m/s
                trace.data *= 1.e-2
            else:
                stream.remove(trace)

    for stream in data_sw:
        for trace in stream:
            trace.weight = 1.
            component = trace.meta.channel[-1].upper()

            if component == 'Z':
                filename = '%s/%s.%s.BH.%d' % (path, event_name, stream.id, 4)
                trace.data = obspy.read(filename, format='sac')[0].data

            elif component == 'R':
                filename = '%s/%s.%s.BH.%d' % (path, event_name, stream.id, 2)
                trace.data = obspy.read(filename, format='sac')[0].data

            elif component == 'T':
                filename = '%s/%s.%s.BH.%d' % (path, event_name, stream.id, 0)
                trace.data = obspy.read(filename, format='sac')[0].data

            # convert from cm/s to m/s
            trace.data *= 1.e-2


    return data_bw, data_sw



def get_synthetics_mtuq(dummy_bw, dummy_sw, greens_bw, greens_sw, mt, 
                        Mw=None, apply_shifts=True):

    synthetics_bw = deepcopy(dummy_bw)
    synthetics_sw = deepcopy(dummy_sw)

    for synthetics, greens in ((synthetics_bw, greens_bw), (synthetics_sw, greens_sw)):
        for i in range(len(synthetics)):
            components = []
            for trace in synthetics[i]:
                components += [trace.meta.channel[-1]]
            greens[i].initialize(components)
            dummy = greens[i].get_synthetics(mt)
            for trace in synthetics[i]:
                trace.weight = 1.
                component = trace.meta.channel[-1].upper()
                trace.data = dummy.select(component=component)[0].data

                if apply_shifts:
                    if Mw==None:
                        # what is the seismic moment of the given moment tensor?
                        M0 = np.sqrt(0.5*np.sum(mt[0:3]**2.) + np.sum(mt[3:6]**2.))
                        Mw = (np.log10(M0) - 9.1)/1.5

                    apply_magnitude_dependent_shift(trace, Mw)

    return synthetics_bw, synthetics_sw



def apply_magnitude_dependent_shift(trace, Mw):
    """ This type of time-shift arises from the idiosyncratic way CAP 
      implements source-time function convolution. CAP's "conv" function
      results in systematic magnitude-dependent shifts between origin times
      and arrival times. This is arguably a bug. We include this function
      to allow benchmark comparisons between MTUQ synthetics (which normally
      lack such shifts) and CAP synthetics.
    """
    # the amount of the time-shift is half the sum of the earthquake's rupture
    # time and rise time, as given by relations in the CAP Perl wrapper
    t_offset = (cap_rupture_time(Mw) + cap_rise_time(Mw))/2.

    dt = trace.stats.delta
    nt = int(t_offset/dt)

    # shift trace to right by nt samples
    trace.data[nt:] = trace.data[:-nt]
    trace.data[:nt] = 0.


def compare_cap_mtuq(cap_bw_, cap_sw_, mtuq_bw_, mtuq_sw_, 
                     bw_tol=np.inf, sw_tol=1.e-2, norm=2):
    """ Checks whether CAP and MTUQ synthetics agree within the specified
      tolerances 

      Even with the magnitude-dependent time-shift correction described above,
      CAP and MTUQ synthetics will not match perfectly because the correction
      is made only after tapering

      For body wave windows, the time-shift correction is large relative to the
      window length, and the tapering-related mismatch will usually be
      pronounced. Thus, checking body waves is turned off by default
    """
    # keep track of number of mismatches
    count = 0

    for cap_bw, cap_sw, mtuq_bw, mtuq_sw in zip(
        cap_bw_, cap_sw_, mtuq_bw_, mtuq_sw_):

        if bw_tol < np.inf:
            maxval = 0.
            for bw in mtuq_bw:
                maxval = max(maxval, abs(bw.data).max())

            for bw1, bw2 in zip(cap_bw, mtuq_bw):
                dt = bw1.stats.delta
                e = np.linalg.norm((bw1.data-bw2.data)*dt, norm)
                e *= dt/maxval
                if e > bw_tol:
                    print ((
                        "Discrepancy between CAP and MTUQ synthetics\n"+
                        "  category:  body waves\n"+
                        "  error:     %e\n"+
                        "  threshold: %e\n") %
                        (e, bw_tol))

                    count += 1


        if sw_tol < np.inf:
            maxval = 0.
            for sw in mtuq_sw:
                maxval = max(maxval, abs(sw.data).max())

            for sw1, sw2 in zip(cap_sw, mtuq_sw):
                dt = sw1.stats.delta
                e = np.linalg.norm((sw1.data-sw2.data)*dt, norm)
                e *= dt/maxval 
                if e > sw_tol:
                    print ((
                        "Discrepancy between CAP and MTUQ synthetics\n"+
                        "  category:  surface waves\n"+
                        "  id:        %s\n"+
                        "  error:     %e\n"+
                        "  threshold: %e\n") %
                        (sw1.id, e, sw_tol))

                    count += 1

    if count > 0:
        raise Exception(
            "Discrepancy between CAP and MTUQ synthetics")




