import os.path
from mtuq.util_SPECFEM3D import get_proc_name
from mtuq.util_SPECFEM3D.ibool_reader import DEnquire_Element
from mtuq.util_SPECFEM3D.sgt_reader import DEnquire_SGT
from mtuq.util_SPECFEM3D.xyz_reader import DEnquire_XYZ_GLLs_Element
from mtuq.util_SPECFEM3D.DGridMgr import DPolyMesh
from mtuq.util_SPECFEM3D.interp_tools import DCreate_anchors_xi_eta_gamma, DLagrange_interp_sgt, DLagrange_any3D

import numpy as np


class DSGTMgr(DPolyMesh):
    '''Strain Green's Tensor (SGT) database Manager'''

    def __init__(self, sgt_database_folder, model3D_folder, info_grid_file):
        '''
        :param sgt_database_folder:     The directory to the SGT database.
        :param model3D_folder:          The directory to the 3D background model.
        :param info_grid_file:          The hdf5 file storing the information of pre-computed grids in the 3D model.
        '''

        self.sgt_database_folder = sgt_database_folder
        self.model3D_folder = model3D_folder
        self.idx_element = -1
        super().__init__(info_grid_file)


    def _initial_element_frame(self):
        ''' return the gll information (index, location) at one selected element. '''
        ibool_file = os.path.join(str(self.model3D_folder), str(self.proc_name)+str("_ibool.bin"))
        self.idx_glls = DEnquire_Element(ibool_file,  self.idx_element)

        x_glls, y_glls, z_glls = DEnquire_XYZ_GLLs_Element(self.model3D_folder,
                                                           self.idx_processor,
                                                           self.idx_element)
        self.xyz_glls = np.transpose(np.vstack([x_glls, y_glls, z_glls]))



    def _initial_SGTs_N_station(self):
        '''Return the SGT between origin and station. '''

        dir_string = os.path.join(str(self.sgt_database_folder),
                                     # str(self.station.network),
                                     str(self.station.station),
                                     str(self.proc_name))
        sgt_data_path = dir_string + str("_sgt_data.bin")
        sgt_info_path = dir_string + str("_header.hdf5")

        self.sgts = DEnquire_SGT(sgt_data_path, sgt_info_path, self.idx_glls)



    def interp_sgt_Lagrange(self, xi, eta, gamma):
        '''Using lagrange method to interpolate the SGT at source (in the mesher)'''
        ngll_x = 3
        ngll_y = 3
        ngll_z = 3

        xi_gll, eta_gll, gamma_gll = DCreate_anchors_xi_eta_gamma(ngll_xyz=3)
        h_xi_arr, h_eta_arr, h_gamma_arr = DLagrange_any3D(xi, eta, gamma, xi_gll, eta_gll, gamma_gll)

        self.sgt_interp = DLagrange_interp_sgt(h_xi_arr, h_eta_arr, h_gamma_arr, self.sgts,
                                          ngll_x=ngll_x, ngll_y=ngll_y, ngll_z=ngll_z)


    def get_sgt(self, station, origin):
        '''Acquire SGT. '''

        self.station = station

        if not self.b_polyMesh_initial:
            raise NotImplementedError

        b_depth = False
        try:
            z = origin.depth_in_m
            b_depth = True
        except:
            z = origin.elevation_in_m
            b_depth = False

        _, _, _, _, \
        _, _, _, \
        idx_processor, element_index, \
        xi, eta, gamma = self.find(x=origin.latitude, y=origin.longitude, z=z, n=1, b_depth=b_depth)

        # MUST subtract 1, element_index in the info_grid_file starts from 1.
        idx_element = element_index - 1
        self.idx_processor = idx_processor
        self.idx_element = idx_element
        self.proc_name = get_proc_name(self.idx_processor)
        self._initial_element_frame()
        self._initial_SGTs_N_station()

        # The Lagrange interpolation to get interpolated SGT
        self.interp_sgt_Lagrange(xi, eta, gamma)
        return self.sgt_interp
