#!/usr/bin/env python

import mtuq
import mtuq.dataset.sac
import mtuq.greens_tensor.syngine

from os.path import basename, join
from mtuq.grid_search import DCGridRandom, DCGridRegular
from mtuq.util.util import AttribDict, root


if __name__=='__main__':
    paths = AttribDict({
        'data':    join(root(), 'tests/data/20090407201255351_debug')})

    print 'made it here-1'
    data = mtuq.dataset.sac.reader(paths.data, wildcard='*.[zrt]')
    data.sort_by_distance()

    print 'made it here-2'
    stations  = []
    for stream in data:
        stations += [stream.station]
    origin = data.get_origin()

    generator = mtuq.greens_tensor.syngine.Generator('ak135f_5s')
    greens = generator(stations, origin)

    grid = DCGridRegular(
        npts_per_axis=1,
        Mw=4.5,
        )

    print 'made it here-3'
    for mt in grid:
        greens.get_synthetics(mt)
