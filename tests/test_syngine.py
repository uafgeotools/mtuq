#!/usr/bin/env python3

import os
import requests
import zipfile
from glob import glob
from urllib.request import URLopener

import numpy as np
import obspy


if __name__=='__main__':

    #
    # Try a simple download
    #

    syngine_url = 'http://service.iris.edu/irisws/syngine/1/query?model=ak135f_2s&dt=0.02&greensfunction=1&sourcedistanceindegrees=0.1425432904952431&sourcedepthinmeters=25000&origintime=2009-04-07T20:12:55.000000&starttime=2009-04-07T20:12:55.000000'

    opener = URLopener()
    opener.retrieve(syngine_url, 'download.zip')
    os.remove('download.zip')

    #
    # Check if a sign change is still necessary to the radial component
    # returned by Instaseis/syngine from a Force source
    #

    url = "http://service.iris.washington.edu/irisws/syngine/1/query?model=ak135f_5s&format=saczip&components=ZRE&units=displacement&receiverlatitude=0&receiverlongitude=1&sourcelatitude=0&sourcelongitude=0&sourcedepthinmeters=0&sourceforce=1e12,0,0&nodata=404"

    response = requests.get(url)
    if response.status_code == 200:
        with open("data.zip", "wb") as f:
            f.write(response.content)
        with zipfile.ZipFile("data.zip", "r") as zip_ref:
            zip_ref.extractall(".")

    stream = obspy.read('*.sac')
    for tr in stream:
        if tr.stats.channel == 'MXE':
            E = tr.data
        if tr.stats.channel == 'MXR':
            R = tr.data

    os.remove('data.zip')
    os.remove('Syngine.log')
    for filename in glob('*.sac'):
        os.remove(filename)

    if np.allclose(E, R):
        # Instaseis/syngine now returns the correct sign on the radial component.
        # The workaround implemented by MTUQ may need to be removed
        raise Exception()


