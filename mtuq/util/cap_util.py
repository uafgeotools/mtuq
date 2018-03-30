

import csv
import warnings
from mtuq.util.wavelets import Wavelet


def remove_unused_stations(dataset, filename):
    """ Removes stations marked for exclusion in CAP weight file
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



def parse_weight_file(filename):
    """ Parses CAP-style weight file
    """
    weights = {}
    with open(filename) as f:
        reader = csv.reader(f, delimiter=' ', skipinitialspace=True)
        for row in reader:
            id = '.'.join(row[0].split('.')[1:4])
            weights[id] = [float(w) for w in row[1:]]

    return weights


class Trapezoid(Wavelet):
    """ Trapezoid-like wavelet obtained by convolving two boxes
        Reproduces capuaf:trap.c
    """

    def __init__(self, rise_time=None):
        warnings.warn('wavelets.Trapezoid not yet tested')

        if rise_time:
            self.rise_time = rise_time
        else:
            raise ValueError


    def evaluate(self, t):
        """ Evaluates wavelet at chosen points
        """
        # rather than an anlytical formula, the following numerical procedure
        # defines the trapezoid
        if t1>t2: t1,t2 = t2,t1
        n1 = max(int(round(t1/dt)),1)
        n2 = max(int(round(t2/dt)),1)
        r = 1./(n1+n2)
        y = np.zeros(n1+n2)
        for i in range(1,n1+1):
            y[i] = y[i-1] + r
            y[-i-1] = y[i]
        for i in range(i,n2):
            y[i] = y[i-1]

        # interpolate from numerical grid to the user-supplied points
        y = np.interp(t0,y0,t)



def trapezoid_rise_time(*args, **kwargs):
    #raise NotImplementedError
    return 1.


