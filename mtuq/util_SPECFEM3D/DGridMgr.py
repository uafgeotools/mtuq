import h5py
import numpy as np

GRID_KEYS = ["latitude",
             "longitude",
             "z",
             "depth",
             "utm_x",
             "utm_y",
             "utm_z",
             "slice_index",
             "element_index",
             "xi",
             "eta",
             "gamma"]


class DPolyMesh():
    ''' The information (see GRID_KEYS) of pre-computed grids in the 3D background model. '''

    def __init__(self, file_path):
        '''
        :param file_path: The HDF5 file storing the database
        '''

        self.b_polyMesh_initial = False
        if file_path is None:
            return

        try:
            with h5py.File(file_path, 'r') as f:
                self.mesh_lat           = f[GRID_KEYS[0]][:]
                self.mesh_long          = f[GRID_KEYS[1]][:]
                self.mesh_z             = f[GRID_KEYS[2]][:]    # in meter
                self.mesh_depth         = f[GRID_KEYS[3]][:]    # in meter.
                self.mesh_utm_x         = f[GRID_KEYS[4]][:]
                self.mesh_utm_y         = f[GRID_KEYS[5]][:]
                self.mesh_utm_z         = f[GRID_KEYS[6]][:]
                self.mesh_slice_index   = f[GRID_KEYS[7]][:]
                self.mesh_element_index = f[GRID_KEYS[8]][:]
                self.mesh_xi            = f[GRID_KEYS[9]][:]
                self.mesh_eta           = f[GRID_KEYS[10]][:]
                self.mesh_gamma         = f[GRID_KEYS[11]][:]

            self.n_grid = len(self.mesh_lat)
            self.b_polyMesh_initial = True
        except:
            print("!!! Information of grids not found!")


    def _check(self):
        if self.b_polyMesh_initial is False:
            print("!!! Information of grids not found!")
            return exit(-1)


    def find(self, x, y, z, n=1, mode='LATLONGZ', b_depth=True):
        '''
        Acquire information of N points near the (x, y, z)
         :param x:       Either the latitude or UTMX.
         :param y:       Either the longitude or UTMY.
         :param z:       The depth or elevation in meter. eg: depth: 2000, elevation: -1200
         :param n:       The number of point enquired.
         :return:        The information of the inquired grid points.
        '''

        self._check()

        n = int(n)
        n_point = len(self.mesh_lat)
        distance_arr = np.zeros(n_point)

        if str(mode).upper() == str('LATLONGZ'):
            if b_depth:
                # use depth.
                for i in range(n_point):
                    _dist_H = 111.0 * 1000.0 * np.sqrt(np.square(self.mesh_lat[i] - x) + np.square(self.mesh_long[i]-y))
                    distance_arr[i] = np.sqrt(np.power(_dist_H, 2) + np.power(self.mesh_depth[i] - z, 2))
            else:
                # use elevation.
                for i in range(n_point):
                    _dist_H = 111.0 * 1000.0 * np.sqrt(np.square(self.mesh_lat[i] - x) + np.square(self.mesh_long[i]-y))
                    distance_arr[i] = np.sqrt(np.power(_dist_H, 2) + np.power(self.mesh_z[i] - z, 2))

        elif str(mode).upper() == str('UTM'):
            if b_depth:
                # use depth
                for i in range(n_point):
                    distance_arr[i] = np.sqrt(np.power(self.mesh_utm_x[i] - x, 2)
                                              + np.power(self.mesh_utm_y[i] - y, 2)
                                              + np.power(self.mesh_depth[i] - z, 2))
            else:
                # use elevation.
                for i in range(n_point):
                    distance_arr[i] = np.sqrt(np.power(self.mesh_utm_x[i] - x, 2)
                                              + np.power(self.mesh_utm_y[i] - y, 2)
                                              + np.power(self.mesh_z[i] - z, 2))
        else:
            print("!! Undefined mode!!")
            raise NotImplementedError

        if 1 == n:
            idx = np.argmin(distance_arr)
        else:
            idx = np.argpartition(distance_arr, n)[:n]

        return self.mesh_lat[idx], self.mesh_long[idx], self.mesh_z[idx], self.mesh_depth[idx],\
               self.mesh_utm_x[idx], self.mesh_utm_y[idx], self.mesh_utm_z[idx], \
               self.mesh_slice_index[idx], self.mesh_element_index[idx], \
               self.mesh_xi[idx], self.mesh_eta[idx], self.mesh_gamma[idx]

