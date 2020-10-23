"""
Data misfit module (fast pure Python version)

See ``mtuq/misfit/__init__.py`` for more information
"""

import numpy as np
from mtuq.util.math import correlate, isclose, list_intersect_with_indices
from mtuq.util.signal import get_components, get_time_sampling



def misfit(data, greens, sources, norm, time_shift_groups,
    time_shift_min, time_shift_max, msg_handle):
    """
    Data misfit function (fast pure Python version)

    See ``mtuq/misfit/__init__.py`` for more information
    """
    helpers = []
    values = np.zeros((len(sources), 1))

    #
    # initialize Green's function machinery
    #
    for _j, d in enumerate(data):
        greens[_j]._set_components(get_components(d))
        helpers += [Helper(data[_j], greens[_j], norm, 
                           time_shift_min, time_shift_max)]

    #
    # iterate over sources
    #
    for _i, source in enumerate(sources):
        source = source.as_vector()

        # optional progress message
        msg_handle()

        #
        # iterate over stations
        #
        for _j, d in enumerate(data):
            components = greens[_j].components
            if not components:
                continue

            # time sampling scheme
            npts, dt = get_time_sampling(d)
            padding_left = int(+time_shift_max/dt)
            padding_right = int(-time_shift_min/dt)


            for group in time_shift_groups:
                # Finds the time-shift between data and synthetics that yields
                # the maximum cross-correlation value across all components in 
                # a given group, subject to min/max constraints
                _, indices = list_intersect_with_indices(
                    components, group)

                ic = helpers[_j].get_time_shift(source, indices)

                for _k in indices:
                    value = 0.

                    if norm=='L1':
                        start = ic
                        stop = start + npts

                        value = dt * helpers[_j].get_L1_norm(
                            source, _k, start, stop)

                    elif norm=='L2':
                        value = dt * helpers[_j].get_L2_norm(
                            source, _k, ic)

                    elif norm=='hybrid':
                        value = dt * helpers[_j].get_L2_norm(
                            source, _k, ic)**0.5

                    try:
                        values[_i] += d[_k].weight * value
                    except:
                        values[_i] += value

    return values



class Helper(object):
    """ 
    Stores data and Green's functions for a given station and provides very
    efficient numerical implementation of L2 norm
    """

    def get_L2_norm(self, source, index, it):
        """
        Given source and component, calculates L2 norm of data and shifted 
        synthetics using

        ||s - d||^2 = s^2 + d^2 - 2sd
        """
        misfit = 0

        # d^2 contribution
        misfit += self.d_d[index]

        # s^2 contribution
        _  = np.dot(self.g_g[index, it, :, :], source)
        misfit += np.dot(np.dot(self.g_g[index, it, :, :], source), source)

        # -2sd contribution
        misfit -= 2*np.dot(self.g_d[index, :, it], source)

        if self.debug:
            synthetics = self.get_synthetics(source)[index]

            dd = np.sum(data**2)
            #print 'error dd:',\
            #    (dd - data_data[index])/dd
            print('dd:',dd)

            ss = np.sum(synthetics**2)
            #print 'error ss:',\
            #    (ss - np.dot(np.dot(greens_greens[index, it, :, :], source), source))/ss
            print('ss:',ss)

            sd = np.sum(synthetics*data)
            #print 'error sd:',\
            #    (sd - np.dot(greens_data[index, :, it], source))/sd
            print('sd:',sd)

            print('')

        return misfit


    def get_L1_norm(self, source, index, start, stop):
        """
        Given source and component, calculates L2 norm of data and shifted 
        synthetics using
        """
        synthetics = self.get_synthetics(source)

        return np.sum(np.abs(
            synthetics[index][start:stop] - 
            self.data[index][start:stop]))


    def get_synthetics(self, source):
        if source != self.source:
            self.source = source
            self.synthetics = self.greens.get_synthetics(source)
        return self.synthetics


    def get_time_shift(self, source, indices):
        """ 
        Finds optimal time shift between the given data and synthetics
        generated from the given source
        """
        cc = self.g_d

        cc_sum = self.cc_sum
        cc_sum[:] = 0.

        for _i in indices:
            cc_sum += np.dot(source, cc[_i, :, :])

        return cc_sum.argmax()


    def __init__(self, d, g, norm, time_shift_min, time_shift_max, debug=False):
        """ 
        Computes auto- and cross-correlations between data and synthetics
        for use by the other two methods
        """
        components = get_components(d)
        if not components:
            return

        self.norm = norm
        self.time_shift_max = time_shift_max
        self.debug = debug

        npts, dt = get_time_sampling(d)

        self.padding_left = int(round(+time_shift_max/dt))
        self.padding_right = int(round(-time_shift_min/dt))
        npts_padding = self.padding_left+self.padding_right

        ncomp = len(components)

        greens = g._array
        ngreens = greens.shape[1]

        self.source = None
        self.cc_sum = np.zeros(npts_padding+1)


        #
        # correlate greens and data
        #
        corr = np.zeros((ncomp, ngreens, npts_padding+1))

        # the main work starts now
        for _i, component in enumerate(g.components):
            trace = d.select(component=component)[0]

            for _j in range(ngreens):
                corr[_i, _j, :] =\
                    correlate(greens[_i, _j, :], trace.data)

        self.g_d = corr


        #
        # autocorrelate data
        #
        corr = np.zeros(ncomp)

        # the main work starts now
        for _i1, trace in enumerate(d):
            corr[_i1] = np.dot(trace.data, trace.data)

        self.d_d = corr


        #
        # autocorrelate greens
        #
        npts, dt = get_time_sampling(g)
        ones = np.pad(np.ones(npts-npts_padding), npts_padding, 'constant')
        corr = np.zeros((ncomp, npts_padding+1, ngreens, ngreens))

        # the main work starts now
        for _i in range(ncomp):
            for _j1 in range(ngreens):
                for _j2 in range(ngreens):

                    if _j1<=_j2:
                        # calculate upper elements
                        corr[_i, :, _j1, _j2] = correlate(
                            greens[_i, _j1, :]*greens[_i, _j2, :], ones)

                    else:
                        # fill in lower elements by symmetry
                        corr[_i, :, _j1, _j2] = corr[_i, :, _j2, _j1]

        self.g_g  = corr


