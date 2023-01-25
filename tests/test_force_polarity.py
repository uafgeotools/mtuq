#!/usr/bin/env python3

import unittest
import numpy as np
import requests
import zipfile
from obspy import read
import os

class TestSyngineForcePolarity(unittest.TestCase):
    """ Test that the force response is correct. MTUQ currently contains 
    a fix for the polarity error in the Radial coordinate. If this test
    fails, the fix may need to be updated. It currently expects the
    polarity of E and R to be opposite.
    
    This test is based on the URL Builder: syngine v.1 
    http://service.iris.washington.edu/irisws/syngine/docs/1/builder/
    """

    def test_force_response(self):
        url = "http://service.iris.washington.edu/irisws/syngine/1/query?model=ak135f_5s&format=saczip&components=ZRE&units=displacement&receiverlatitude=0&receiverlongitude=1&sourcelatitude=0&sourcelongitude=0&sourcedepthinmeters=0&sourceforce=1e12,0,0&nodata=404"

        response = requests.get(url)

        if response.status_code == 200:
            with open("data.zip", "wb") as f:
                f.write(response.content)

            with zipfile.ZipFile("data.zip", "r") as zip_ref:
                zip_ref.extractall(".")

        components = 'ZRE'
        # The format of sac files is the following XX.S0001.SE.MX[component].sac
        # where XX is the network code, S0001 is the station code, SE is the
        # location code, MX is the channel code, and [component] is the
        # component code. The component code is one of Z, R, E

        # Read the sac files
        st = read('*.sac')

        # Get the data for E and R
        for tr in st:
            if tr.stats.channel == 'MXE':
                E = tr.data
            if tr.stats.channel == 'MXR':
                R = tr.data

        # The polarity of E and R should be opposit. 
        # Display a custom error message if this is not the case.
        self.assertTrue(np.allclose(E, R), msg='The polarity of E and R is not opposite. The fix for the polarity error in the Radial coordinate may need to be updated.')

        # Remove all sac files, the Syngine.log and the data.zip
        os.remove('data.zip')
        os.remove('Syngine.log')
        for tr in st:
            os.remove(tr.stats.network+'.'+tr.stats.station+'.'+tr.stats.location+'.'+tr.stats.channel+'.sac')

if __name__ == '__main__':
    unittest.main()