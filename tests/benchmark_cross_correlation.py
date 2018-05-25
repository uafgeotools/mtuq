

from scipy.signal import fftconvolve
import numpy as np
from mtuq.util.util import timer2



@timer2
def correlate_fd(w1,w2):
    fftconvolve(w1,w2[::-1],'valid')


@timer2
def correlate_td(w1,w2):
    np.convolve(w1,w2,'valid')



if __name__=='__main__':
    """
    Compares computational cost of two different implementations of 
    cross-correlation. More precisely, measures how execution time varies as a
    function of vector length and maximum lag time for time-domain and 
    frequency-domain implementations of cross-correlation
    """
    # what vector lengths will we consider? 
    # i.e. what values of len(w11)
    u1 = range(100,10000,100)
    n1 = len(u1)

    # what maxmimum lag times will we consider?
    # i.e. what values of len(w2) - len(w1)
    u2 = range(10,10000,10)
    n2 = len(u2)

    # preallocate vectors
    v1 = {}
    for n in u1:
        w1[n] = np.ones(n)

    # preallocate array for execution time results
    results = np.ones((n1*n2,5))

    k = 0
    for i1 in u1:
        print i1
        for i2 in u2:
            w1_ = w1[i1]
            w2_ = np.ones(i1+i2)

            # keep track of vector length
            results[k,0] = i1

            # keep track of maximum lag time
            results[k,1] = i2

            # measure execution time
            results[k,2] = correlate_fd(w1,w2)
            results[k,3] = correlate_td(w1,w2)

            k+=1

    if 1==1:
        # save execution time results
        np.savetxt('benchmark_correlate_w1_w2_FD_TD',results)

    if 1==1:
        # plot executation time results
        x=results[:,0]
        y=results[:,1]
        z1=results[:,2]
        z2=results[:,3]

        xmin = x.min()
        xmax = x.max()
        ymin = y.min()
        ymax = y.max()

        X = np.linspace(xmin,xmax,1000.)
        Y = np.linspace(ymin,ymax,1000.)

        X,Y,Z1 = mesh2grid(z1, x, y)
        X,Y,Z2 = mesh2grid(z2, x, y)

        pylab.pcolor(X,Y,Z1)
        pylab.savefig('benchmarks_correlate_FD.png')

        pylab.pcolor(X,Y,Z2)
        pylab.savefig('benchmarks_correlate_TD.png')

