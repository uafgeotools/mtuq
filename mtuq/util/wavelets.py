

import numpy as np


def trapezoid(rise_time=None, delta=None):
    if not rise_time:
        raise ValueError

    if not delta:
        raise ValueError

    return _trapezoid(2*rise_time, rise_time, delta)


def _trapezoid(t1,t2,dt):
    """ Convolves two boxes to get trapezoid-like wavelet

        Reproduces capuaf:trap.c
    """
    if t1>t2: t1,t2 = t2,t1
    n1 = _round(t1/dt)
    n2 = _round(t2/dt)
    r = 1./(n1+n2)
    s = np.zeros(n1+n2)
    for i in range(1,n1+1):
        s[i] = s[i-1] + r
        s[-i-1] = s[i]
    for i in range(i,n2):
        s[i] = s[i-1]
    return s


def _round(x):
    return max(int(round(x)),1)
