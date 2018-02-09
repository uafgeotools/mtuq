
import numpy as np

def resample(trace, t1, t2, dt):
    """ 
    t1: desired start time for resampled trace
    t2: desired end time for resampled trace
    dt: desired time increment for resampled trace    
    """
    if trace.stats.delta != dt:
        raise NotImplementedError

    # start and end times prior to resampling
    tmp1 = float(trace.stats.starttime)
    tmp2 = float(trace.stats.endtime)

    # handle roundoff
    d1 = round(tmp1/dt)*dt-tmp1
    d2 = round((tmp2-tmp1)/dt)*dt
    tmp1 += d1
    tmp2 = tmp1 + d2

    # offset between new and old start times
    i1 = int(round((tmp1-t1)/dt))

    # offset between new and old end times
    i2 = int(round((tmp2-t2)/dt))

    # allocate array
    nt = int(round((t2-t1)/dt))
    resampled_data = np.zeros(nt+1)

    #FIXME: NEED TO DOUBLE CHECK ALL CASES
    # cut both ends
    if tmp1 <= t1 <= t2 <= tmp2:
        resampled_data[0:nt] = trace.data[i1:i2]

    # cut beginning only
    elif tmp1 <= t1 <= tmp2 <= t2:
        resampled_data[0:nt-i2] = trace.data[i1:]

    # cut end only
    elif t1 <= tmp1 <= t2 <= tmp2:
        resampled_data[i1:nt] = trace.data[0:i2]

    # cut neither end
    elif t1 <= tmp1 <= tmp2 <= t2:
        resampled_data[i1:i2] =  trace.data[0:]

    return resampled_data


def convolve(u, v):
    return np.convolve(u, v, mode='same')

