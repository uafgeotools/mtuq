#!/usr/bin/env python


import numpy as np
import unittest

from os.path import join
from mtuq.dataset.sac import\
    reader
from mtuq.greens_tensor.syngine import\
    download_greens_tensor, download_synthetics,\
    GreensTensor, GreensTensorFactory
from mtuq.grid_search import MTGridRandom, MTGridRegular
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
        syn = greens.get_synthetics(mt)

        # download reference synthetics
        filename = download_synthetics(model, station, origin, mt)
        dirname = unzip(filename)
        ref = reader(dirname)

        for component in ['Z', 'R', 'T']:
             s = syn[0].select(component=component)[0]
             r = ref[0].select(component=component)[0]

             np.testing.assert_allclose(
                 s.data,
                 r.data,
                 rtol=1E-3, atol=1E-10)




    def get_moment_tensor(self):
        return [1.04e22,-0.039e22,-1e22,0.304e22,-1.52e22,-0.119e22]
        #return MTGridRegular(npts_per_axis=1, Mw=1.).get(0)



    def get_moment_tensor_randome(self):
        return MTGridRandom(npts=1, Mw=1.).get(0)


    def get_station(self):
        return self.get_data().get_station()


    def get_origin(self):
        return self.get_data().get_origin()


    def get_data(self):
        if not hasattr(self, '_data'):
            path = join(path_mtuq(), 'tests/data/20090407201255351')
            data = reader(path, wildcard='*.[zrt]')
            data.sort_by_distance()
            self._data = data
        return self._data

       


if __name__=='__main__':
    unittest.main()

