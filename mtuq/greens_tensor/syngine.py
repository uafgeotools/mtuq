
import obspy
import numpy as np

from mtuq.greens_tensor.axisem_netcdf import GreensTensor as GreensTensorBase


class GreensTensor(GreensTensorBase):
    """
    Adds syngine capabilities to AxiSEM base class

    Syngine is an webservice that provides Green's functions and synthetic
    seismograms for download as compressed SAC files. 

    Syngine uses precomputed AxiSEM databases under the hood, so Green's 
    function conventions, moment tensor conventions, and so on are very similar to
    AxiSEM, and it is not necessary to modify any of the machinery for 
    generating synthetics.
    """
    pass


