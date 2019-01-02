#!/usr/bin/env python


import numpy as np
import unittest
import matplotlib.pyplot as pyplot


from os.path import join
from mtuq.dataset.sac import\
    reader
from mtuq.greens_tensor.syngine import\
    download_greens_tensor, download_synthetics, get_synthetics_syngine,\
    GreensTensor, GreensTensorFactory
from mtuq.grid_search import FullMomentTensorGridRandom, FullMomentTensorGridRegular
from mtuq.util.util import AttribDict, path_mtuq, unzip

 
class greens_tensor_syngine(unittest.TestCase):
    def test_download_greens_tensor(self):
        model = 'ak135f_2s'
        station = self.get_station()
        origin = self.get_origin()

        filename = download_greens_tensor(model, station, origin)
        dirname = unzip(filename)


    def test_download_synthetics(self):
        model = 'ak135f_2s'
        delta = 0.02
        station = self.get_station()
        origin = self.get_origin()
        mt = self.get_moment_tensor()

        filename = download_synthetics(model, station, origin, mt)
        dirname = unzip(filename)


    def test_generate_synthetics(self):
        station = self.get_station()
        origin = self.get_origin()
        mt = self.get_moment_tensor()

        # generate synthetics
        model = 'ak135f_2s'
        factory = GreensTensorFactory(model)
        greens = factory(station, origin)
        syn = greens.get_synthetics(mt)[0]

        # download reference synthetics
        ref = get_synthetics_syngine(model, station, origin, mt)

        for component in ['Z', 'R', 'T']:
             r = ref.select(component=component)[0]
             s = syn.select(component=component)[0]

             if True:
                 npts = s.stats.npts
                 t = np.arange(npts)
                 pyplot.plot(t, s.data, t, r.data)
                 pyplot.show()
                 pyplot.close()
                 #pyplot.savefig('syngine.png')

             np.testing.assert_allclose(
                 s.data,
                 r.data,
                 rtol=1.e-3,
                 atol=0.25*r.data.max())


    def get_moment_tensor(self):
        return [1.04e22,-0.039e22,-1e22,0.304e22,-1.52e22,-0.119e22]
        #return 1.e15*np.array([-2.449e-01, 3.400e-01, -9.507e-02,
        #                        8.821e-01,-3.542e-01,  6.440e-02]) # north slope event

    def get_moment_tensor_random(self):
        return FullMomentTensorGridRandom(npts=1, Mw=1.).get(0)


    def get_station(self):
        return self.get_data().get_station()


    def get_origin(self):
        return self.get_data().get_origin()


    def get_data(self):
        if not hasattr(self, '_data'):
            path = join(path_mtuq(), 'data/examples/20090407201255351')
            data = reader(path, wildcard='*.[zrt]')
            data.sort_by_distance()
            self._data = data
        return self._data

       


if __name__=='__main__':
    unittest.main()

