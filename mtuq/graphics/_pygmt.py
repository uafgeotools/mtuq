
#
# graphics/big_beachball.py - first motion "beachball" plots with piercing point polarities
#


import os
import matplotlib.pyplot as pyplot
import numpy as np

from mtuq.util import warn


def exists_pygmt():
    try:
        import pygmt
        return True
    except:
        return False


