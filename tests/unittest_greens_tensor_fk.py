#!/usr/bin/env python


import numpy as np
import unittest
import obspy.core
from mtuq.dataset.base import DatasetBase as Dataset
from mtuq.misfit import cap
from mtuq.util import AttribDict
from mtuq.util.wavelets import Gaussian

 
class TestFK(unittest.TestCase):
    def test_generate_synthetics1(self):
        """ Checks that generate_synthetics matches a different python 
            implementation based closely on CAP
        """
        pass


    def test_generate_synthetics2(self):
        """ Checks that generate_synthetics output matches precomputed 
            CAP synthetics for the 2009 example
        """
        pass

       

### utility functions

def Stream(*args, **kwargs):
    """ Overloads obspy Stream by seeting the "id" attribute, which 
        mtuq expects (normally this is done by dataset.reader)
    """
    stream = obspy.core.Stream(*args, **kwargs)
    stream.id = 0
    return stream


def Trace(*args, **kwargs):
    """ Overloads obspy Trace by seeting the "weight" attribute, which 
        cap.misfit expects (normally this is done by cap.process_data)
    """
    trace = obspy.core.Trace(*args, **kwargs)
    trace.weight = 1.
    return trace


if __name__=='__main__':
    unittest.main()

