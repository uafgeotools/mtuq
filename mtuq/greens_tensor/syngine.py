
import obspy
import numpy as np

from mtuq.greens_tensor.axisem_netcdf import GreensTensor as GreensTensorBase


class GreensTensor(GreensTensorBase):
    """
    Syngine Green's tensor object

    Syngine uses precomputed AxiSEM databases under the hood, so Green's 
    function conventions, moment tensor conventions, and so on are very similar,
    and it is not currently necessary to modify any of the machinery for
    generating synthetics
    """
    pass


