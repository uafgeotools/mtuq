
import numpy as np

def resample(trace, starttime, endtime, dt):
        t1 = trace.stats.starttime
        t2 = trace.stats.endtime

        if trace.stats.delta != dt:
            raise NotImplementedError

        # offsets
        i1 = int((starttime-t1)/dt)
        i2 = int((endtime-t2)/dt)

        # allocate array
        nt = int((endtime-starttime)/dt)
        resampled_data = np.zeros(nt+1)

        # cut both ends
        if t1 <= starttime <= endtime <= t2:
            resampled_data[0:nt] = trace.data[i1:i2]

        # cut beginning only
        elif t1 <= starttime <= t2 <= endtime:
            resampled_data[0:nt-i2] = trace.data[i1:]

        # cut end only
        elif starttime <= t1 <= endtime <= t2:
            resampled_data[i1:nt] = trace.data[0:-i2]

        # cut neither end
        elif starttime <= t1 <= t2 <= endtime:
            resampled_data[i1:nt-i2] =  trace.data[0:]

        return resampled_data


def convolve(u, v):
    return np.convolve(u, v, mode='same')

