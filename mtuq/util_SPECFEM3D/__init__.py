## Set parameters based on the 3D background model. ##################
#
# Constant parameters for the 3D background model.
# Number of spectral element per slice 
NSPEC_PER_SLICE = 2232
#
# The directory to 3D background model files, "*/DATABASES_MPI/"
MODEL3D_folder = "*/DATABASES_MPI/"


## Set parameters based on the Strain Green's Tensor database. #######
#
#Sampling rate
SAMPLING_RATE = 2
DT = 1.0/SAMPLING_RATE 
#
# The directory to SGT database. eg: */SGT/
SGT_DATABASE_folder = "*/SGT/"


## Set the file path to the hdf5 file that stores the pre-computed ###
## information of user-defined grid points in the 3D model.        ###
INFO_GRID_file = "*/InfoGrid.hdf5"


############### DO NOT CHANGE ################################
SGT_ENCODING_LEVEL = 8
NGLL_SPEC_COMPRESSION = 27

NGLLX = 5
NGLLY = NGLLX
NGLLZ = NGLLX
CONSTANT_INDEX_27_GLL = [0, 2, 4, 10, 12, 14, 20, 22, 24,
                         50, 52, 54, 60, 62, 64, 70, 72, 74,
                         100, 102, 104, 110, 112, 114, 120, 122, 124]


def get_proc_name(idx_processor):
    '''Return the processor name.'''
    return str('proc')+str(idx_processor).rjust(6, '0')
    
