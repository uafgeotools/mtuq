
import numpy as np
from copy import copy
from mtuq.util.math import isclose
from obspy.geodetics import gps2dist_azimuth, kilometers2degrees
from obspy.signal.filter import highpass, lowpass
from scipy.signal import fftconvolve


def cut(trace, t1, t2):
    """ 
    trace: obspy trace
    t1: desired start time
    t2: desired end time
    """
    if t1 < float(trace.stats.starttime):

        _debug_msg(trace, t1, t2)

        raise Exception('The chosen window begins before the trace.  Consider '
           'using a later window, or to automatically pad the beginning of the '
           'trace with zeros, use mtuq.util.signal.resample instead')

    if t2 > float(trace.stats.endtime):

        _debug_msg(trace, t1, t2)

        raise Exception('The chosen window ends after the trace.  Consider '
           'using an earlier window, or to automatically pad the end of the '
           'trace with zeros, use mtuq.util.signal.resample instead')

    t0 = float(trace.stats.starttime)
    dt = float(trace.stats.delta)
    it1 = int((t1-t0)/dt)
    it2 = int((t2-t0)/dt)
    trace.data = trace.data[it1:it2]
    trace.stats.starttime = t1
    trace.stats.npts = it2-it1


def resample(data, t1_old, t2_old, dt_old, t1_new, t2_new, dt_new):
    """ 
    data: numpy array
    t1_new: desired start time for resampled data
    t2_new: desired end time for resampled data
    dt_new: desired time increment for resampled data
    """
    dt = dt_old

    tmp1 = round(t1_old/dt)*dt
    tmp2 = round((t2_old-t1_old)/dt)*dt
    t1_old = tmp1
    t2_old = tmp1 + tmp2

    tmp1 = round(t1_new/dt)*dt
    tmp2 = round((t2_new-t1_new)/dt)*dt
    t1_new = tmp1
    t2_new = tmp1 + tmp2

    # offset between new and old start times
    i1 = int(round((t1_old-t1_new)/dt))

    # offset between new and old end times
    i2 = int(round((t2_old-t2_new)/dt))

    nt = int(round((t2_new-t1_new)/dt))
    adjusted = np.zeros(nt+1)

    #
    # adjust end points, leaving sampling rate unchaged for now
    #

    if t1_old <= t1_new <= t2_new <= t2_old:
        # cut both ends
        adjusted[0:nt] = data[-i1:nt-i1]

    elif t1_old <= t1_new <= t2_old <= t2_new:
        # cut left, pad right
        adjusted[0:nt+i2] = data[-i1:]

    elif t1_new <= t1_old <= t2_new <= t2_old:
        # pad left, cut right
        adjusted[i1:nt] = data[:-i2-1]

    elif t1_new <= t1_old <= t2_old <= t2_new:
        # pad both ends
        adjusted[i1:i2] =  data[:]

    #
    # adjust sampling rate
    #

    nt_old = nt
    nt_new = int(round((t2_new-t1_new)/dt_new))

    if nt_new==nt_old:
        return adjusted
    elif dt_new > dt_old:
        return downsample(adjusted, dt_old, dt_new, nt_old, nt_new)
    else:
        return upsample(adjusted, dt_old, dt_new, nt_old, nt_new)


def _resample_trace(data, dt_old, dt_new):
    # sometimes returns a result that is one sample too short
    from obspy.core.trace import Trace
    trace = Trace(data, {'npts': len(data), 'delta':dt_old})
    trace.resample(dt_new**-1)
    return trace.data


def downsample(data, dt_old, dt_new, nt_old, nt_new):
    freq = dt_new**-1

    freq_nyquist = 0.5*dt_old**-1

    if freq > 0.999*freq_nyquist:
        freq = 0.999*freq_nyquist

    filtered = lowpass(data, freq=freq, df=dt_old**-1, zerophase=True)

    t1, t2 = 0., nt_new*dt_new
    t_old = np.linspace(t1, t2, nt_old+1)
    t_new = np.linspace(t1, t2, nt_new+1)

    return np.interp(t_new, t_old, filtered)


def upsample(data, dt_old, dt_new, nt_old, nt_new):
    t1, t2 = 0., nt_new*dt_new
    t_old = np.linspace(t1, t2, nt_old+1)
    t_new = np.linspace(t1, t2, nt_new+1)
    return np.interp(t_new, t_old, data)


def pad(trace, padding):
    dt = trace.stats.delta

    npts_padding = (
        int(round(abs(padding[0])/dt)),
        int(round(abs(padding[1])/dt)),
        )

    trace.data = np.pad(trace.data, npts_padding, 'constant')
    trace.stats.starttime += abs(padding[0])


def check_time_sampling(stream):
    """ Checks if all traces in stream have the same time sampling
    """
    starttime = [float(trace.stats.starttime) for trace in stream]
    starttime0 = [float(stream[0].stats.starttime)]*len(stream)
    if not isclose(starttime, starttime0):
        return False

    delta = [trace.stats.delta for trace in stream]
    delta0 = [stream[0].stats.delta]*len(stream)
    if not isclose(delta, delta0):
        return False

    npts = [trace.stats.npts for trace in stream]
    npts0 = [stream[0].stats.npts]*len(stream)
    if npts != npts0:
        return False

    return True


def check_padding(dataset, time_shift_min, time_shift_max):
    # For greatest accuracy, Green's functions must be longer than data
    # (i.e., Green's functions must begin +time_shift_max before data and
    # end -time_shift_min after data.) 
    #
    # To achieve this, users can supply left and right padding lengths
    # via the `padding` parameter when calling `mtuq.ProcessData`.
    #
    # If padding lengths were not supplied, then Green's functions will be
    # padded with zeros, which may result in less accurate time shifts

    for stream in dataset:
        _, dt = get_time_sampling(stream)

        for trace in stream:
            npts_left = getattr(trace, 'npts_left', 0)
            npts_right = getattr(trace, 'npts_right', 0)

            if (time_shift_min!=0 or time_shift_max!=0) and\
               npts_left==0 and npts_right==0:

                pad(trace, (time_shift_min, time_shift_max))

                setattr(trace, 'npts_left', int(round(+time_shift_max/dt)))
                setattr(trace, 'npts_right', int(round(-time_shift_min/dt)))

            else:
                assert npts_left == int(round(+time_shift_max/dt))
                assert npts_right == int(round(-time_shift_min/dt))


def get_arrival(arrivals, phase):
    phases = []
    for arrival in arrivals:
        phases += [arrival.phase.name]

    if phase not in phases:
        raise Exception("Phase not found")

    arrival = arrivals[phases.index(phase)]
    return arrival.time


def check_components(stream):
    """ Raises an exception if multiple components of same type found in stream
    """
    components = set()
    for component in get_components(stream):
        if component in components:
            try:
                print('trace.id:', stream.select(component=component)[0].id)
            except:
                pass
            raise Exception('Multiple %s components in stream' % component)
        components.add(component)


def get_components(stream):
    components = []
    for trace in stream:
        components += [trace.stats.channel[-1].upper()]
    return components


def get_distance_in_m(station, origin):
    distance_in_m, _, _ = gps2dist_azimuth(
        origin.latitude,
        origin.longitude,
        station.latitude,
        station.longitude)
    return distance_in_m


def get_distance_in_deg(station, origin):
    return m_to_deg(
        get_distance_in_m(station, origin))


def get_time_sampling(stream):
    if len(stream) > 0:
        npts = stream[0].data.size
        dt = stream[0].stats.delta
    else:
        npts = None
        dt = None
    return npts, dt


def m_to_deg(distance_in_m):
    return kilometers2degrees(distance_in_m/1000., radius=6371.)


def isempty(dataset):
    if dataset is None:
        return True
    elif len(dataset)==0:
        return True
    elif sum([len(stream) for stream in dataset])==0:
        return True
    else:
        return False


def _debug_msg(trace, t1, t2):
        #
        # can be useful for debugging cut() and resample() exceptions and
        # other temporal out-of-bounds errors
        #
        print()
        print(trace.id)
        print('starttime:', trace.stats.starttime)
        print('endtime:  ', trace.stats.endtime)
        print()
        print('starttime (float):', float(trace.stats.starttime))
        print('endtime (float):  ', float(trace.stats.endtime))
        print()
        print('t1:', t1)
        print('t2:', t2)
        print()



def _window_warnings(window_type, window_length):
    if window_type=='minmax':
        print(
              '  \n'
              '  WARNING:'
              '  \n'
              '  \n'
              '  Experimental distance-dependent window lengths may not be \n'
              '  supported by all MTUQ functions.'
              '  \n'
             )

        if not window_length:
            print(
                  '  A minimum window length of %f s will be enforced.'
                  '  \n'
                  % window_length
                  )
        else:
            print(
                  '  No minimum window length will be enforced.'
                  '\n'
                  )

