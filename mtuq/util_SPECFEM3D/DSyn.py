from obspy.core.stream import Stream
from obspy.core.trace import Trace

import numpy as np

def RTP_to_DENZ(mt):
    ''' MT from RTP to ENZ   '''
    new_mt = np.zeros_like(mt)
    new_mt[0] = mt[2]
    new_mt[1] = mt[1]
    new_mt[2] = mt[0]
    new_mt[3] = -1.0 * mt[5]
    new_mt[4] = mt[4]
    new_mt[5] = -1.0 * mt[3]
    return new_mt


def DSyn(mt, sgt, element):
    '''
    :param mt:      The moment tensor (MT) in ENZ
    :param SGT:     The strain Green's tensor.
                    The force order: N-E-Z
    :param element: The string of MT component.
    :return:
    '''

    new_mt = mt.copy()
    new_mt[3:] = 2.0 * new_mt[3:]

    n_force = 3
    n_element = 6
    stream = Stream()
    channels = ['N', 'E', 'Z']
    for i in range(n_force):
        trace = Trace(np.dot(sgt[:, i, :], mt))
        trace.stats.channel = str(element)+str(channels[i])
        stream.append(trace)
    return stream
