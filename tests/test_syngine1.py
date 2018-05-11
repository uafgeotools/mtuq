#!/usr/bin/env python

import urllib
from mtuq.util.util import unzip


def download(model, distance, depth):
    """ Downloads Green's function from syngine through URL interface
    """
    url = ('http://service.iris.edu/irisws/syngine/1/query'
         +'?model='+model
         +'&greensfunction=1'
         +'&sourcedistanceindegrees='+str(distance)
         +'&sourcedepthinmeters='+str(int(round(depth))))
    filename = ('syngine-'
         +'model='+model
         +'&greensfunction=1'
         +'&sourcedistanceindegrees='+str(distance)
         +'&sourcedepthinmeters='+str(int(round(depth)))
         +'.zip')
    download = urllib.URLopener()
    download.retrieve(url, filename)

    return filename


if __name__=='__main__':
    model = 'ak135f_5s'
    distance = 10.5
    depth = 1000
    filename = download(model, distance, depth)
    unzip(filename)
