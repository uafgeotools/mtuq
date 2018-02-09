
import numpy as np

def convolve(u, v):
    return np.convolve(u, v, mode='same')



def resample(data, t1_old, t2_old, dt_old, t1_new, t2_new, dt_new):
    """ 
    t1_new: desired start time for resampled data
    t2_new: desired end time for resampled data
    dt_new: desired time increment for resampled data
    """
    if dt_old != dt_new:
        raise NotImplementedError
    dt = dt_new

    # correct roundoff
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
    resampled_data = np.zeros(nt+1)

    #FIXME: NEED TO DOUBLE CHECK ALL CASES
    # cut both ends
    if t1_old <= t1_new <= t2_new <= t2_old:
        resampled_data[0:nt] = data[i1:i2]

    # cut beginning only
    elif t1_old <= t1_new <= t2_old <= t2_new:
        resampled_data[0:nt+i2] = data[i1:]

    # cut end only
    elif t1_new <= t1_old <= t2_new <= t2_old:
        resampled_data[i1:nt] = data[0:i2]

    # cut neither end
    elif t1_new <= t1_old <= t2_old <= t2_new:
        resampled_data[i1:i2] =  data[0:]

    return resampled_data


