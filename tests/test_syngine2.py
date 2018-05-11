#!/usr/bin/env python

import mtuq
import mtuq.dataset.sac
import mtuq.greens_tensor.syngine

from os.path import basename, join
from mtuq.util.util import AttribDict, root


if __name__=='__main__':
    paths = AttribDict({
        'data':    join(root(), 'tests/data/20090407201255351_debug')})

    data = mtuq.dataset.sac.reader(paths.data, wildcard='*.[zrt]')
    data.sort_by_distance()

    stations  = []
    for stream in data:
        stations += [stream.station]
    origin = data.get_origin()

    generator = mtuq.greens_tensor.syngine.Generator('ak135f_5s')
    greens = generator(stations, origin)

