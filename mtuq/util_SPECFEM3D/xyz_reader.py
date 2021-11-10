import os.path
from mtuq.util_SPECFEM3D import get_proc_name, CONSTANT_INDEX_27_GLL
from mtuq.util_SPECFEM3D.ibool_reader import read_ibool_by_scipy

from scipy.io import FortranFile


def read_xyz_bin_by_scipy(file_path):
    '''
    * read XYZ bin file by using the SciPy package.

    :param file_path: The path of the XYZ bin file.
    :param file_type: The data type of the XYZ bin.

    * x,y,z -- float32

    :return: The whole data.
    '''

    data_type = 'float32'

    try:
        f = FortranFile(file_path, 'r')
        dat = f.read_reals(dtype=data_type)
        f.close()
    except:
        print("Unable to open file: ", str(file_path))
        return None
    return dat


def DEnquire_XYZ_GLLs_Element(data_dir, idx_processor, idx_element):
    '''
    return the x, y, z of the 27 GLL points where the SGT been stored in the selected element.

    :param data_dir:        The dir of the *.bin files.
    :param idx_proc:        The index of the processor. INT
    :param idx_element:     The index of the element in the processor. INT
    :param NSPEC_PER_SLICE:    The number of the element in the processor. INT
    :return:                The x, y, and z array of the GLL points.
    '''

    proc_name = get_proc_name(idx_processor)
    ibool_file = os.path.join(str(data_dir),  str(proc_name) + "_ibool.bin")
    x_file = os.path.join(str(data_dir),  str(proc_name) + "_x.bin")
    y_file = os.path.join(str(data_dir), str(proc_name) + "_y.bin")
    z_file = os.path.join(str(data_dir), str(proc_name) + "_z.bin")

    ibool = read_ibool_by_scipy(ibool_file)
    
    x = read_xyz_bin_by_scipy(x_file)
    y = read_xyz_bin_by_scipy(y_file)
    z = read_xyz_bin_by_scipy(z_file)
    glls_idx = ibool[idx_element][CONSTANT_INDEX_27_GLL]

    return x[glls_idx], y[glls_idx], z[glls_idx]

