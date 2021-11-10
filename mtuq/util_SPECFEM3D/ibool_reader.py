from mtuq.util_SPECFEM3D import NGLLX, NGLLY, NGLLZ, CONSTANT_INDEX_27_GLL
from mtuq.util_SPECFEM3D import NSPEC_PER_SLICE
from scipy.io import FortranFile
import numpy as np


def read_ibool_by_scipy(ibool_file):
    '''
        Read the ibool file in the folder of 3D background model.
    '''

    f = FortranFile(ibool_file, 'r')
    ibool = f.read_reals(dtype='int32')
    f.close()
    ibool = np.reshape(ibool, (NSPEC_PER_SLICE, NGLLX * NGLLY * NGLLZ))

    # The index in *.ibool files starts from 1.
    ibool = ibool - 1
    return ibool


def DEnquire_Element(ibool_file, index_element):
    ''' Read the index of the 27 GLL points where the SGT been stored in the selected element.'''

    ibool = read_ibool_by_scipy(ibool_file)
    if ibool.__len__() <= index_element:
        return np.zeros(27)

    else:
        NGLLX_N3 = 3
        NGLLY_N3 = 3
        NGLLZ_N3 = 3

        # The global index in slice of selected GLL points.
        gll_array = ibool[index_element][CONSTANT_INDEX_27_GLL]

        # sort the index.
        gll_points = []
        gll_array = np.reshape(gll_array, [NGLLZ_N3, NGLLY_N3, NGLLX_N3])
        for i in range(NGLLX_N3):
            for j in range(NGLLY_N3):
                for k in range(NGLLZ_N3):
                    gll_points.append(gll_array[k, j, i])
        gll_points = np.asarray(gll_points)

        return gll_points
