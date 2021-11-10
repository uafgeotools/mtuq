import numpy as np

# Lagrange polynomial
def DCreate_anchors_xi_eta_gamma(ngll_xyz):
    '''
    * The xi, eta, and gamma array for the anchor points. (Constant)
    * the index of the location array: [Z, Y, X]

    :param ngll_xyz: The number of GLL point in each direction.
                     5 if all 125 GLL points are selected.
                     3 if the 27 designated GLL points are selected.
    :return:
            the xi, eta, and gamma array.
    '''

    if 3 == int(ngll_xyz):
        xi = np.array([-1.0, 0.0, 1.0])
    else:
        xi = np.array([-1.0, -1.0 * np.sqrt(3.0 / 7.0), 0.0, np.sqrt(3.0 / 7.0), 1.0])
    eta = xi.copy()
    gamma = xi.copy()
    return xi, eta, gamma


def DLagrange_any(xi, xi_gll):
    '''
    * compute the lagrange coefficient for the giving data.

    :param xi:      The data to be interpolated.
    :param x_gll:   The giving data. a set of point. np.array.
    :return: h_arr: The coefficient for the interpolation.
    '''

    nGll = len(xi_gll)
    h_arr = []
    for i in range(nGll):
        prod1 = 1.0
        prod2 = 1.0
        x0 = xi_gll[i]
        for j in range(nGll):
            if j == i:
                continue

            x = xi_gll[j]
            prod1 *= (xi - x)
            prod2 *= (x0 - x)

        h_arr.append(prod1/prod2)
    h_arr = np.asarray(h_arr)
    return h_arr


def DLagrange_any3D(xi, eta, gamma, xi_gll, eta_gll, gamma_gll):
    '''
    * compute the lagrange coefficient for the giving 3D data.

    :param xi: The relative x of the location to be interpolated.
    :param eta: The relative y of the location to be interpolated.
    :param gamma: The relative z of the location to be interpolated.
               The xi, eta, gamma are the relative location.
               Size: 1
               Data type: nd.float
    :param xi_gll: The x array of the knowing gll points.
                  Eg: Data size: NGLL=5.
                  Data type: np.array.
    :param eta_gll: The y array of the knowing points.
                    Eg: Data size: 5.
                    Data type: np.array.
    :param gamma_gll: The z array of the knowing points.
                      Eg: Data size: 5.
                      Data type: np.array.
    :return: coef
    Eg: Data size: 125, represents coefficient of the 125 gll points.
                  Data type: np.array.
    '''

    h_xi_arr = DLagrange_any(xi, xi_gll)
    h_eta_arr = DLagrange_any(eta, eta_gll)
    h_gamma_arr = DLagrange_any(gamma, gamma_gll)
    return h_xi_arr, h_eta_arr, h_gamma_arr


def DLagrange_interp_sgt(h_xi_arr, h_eta_arr, h_gamma_arr, sgt_arr_list, ngll_x, ngll_y, ngll_z):
    '''
    * Utilizing the coef_3d and the value array to interpolation.

    :param h_xi_arr:    The computed xi array for interpolation.
    :param h_eta_arr:   The computed eta array for interpolation.
    :param h_gamma_arr: The computed gamma array for interpolation.
    :param sgt_arr_list: List of the SGT data on the selected GLL points. Typically 125 GLL points.
                    sgt_arr size: 125 * [n_step, 3, 6], where
                    * The 125 represents the number of GLL points in one element.
                    * The sgt order = ibool.
                    * The n_step represents the number of the data for the database.
                    * The 3 represents the number of force.
                    * The 6 represents the number of the element in SGT.
    :param ngll_x:    The number of GLL point in X direction.
                      data type: INT.
                      eg: 5 (NGLLX) if 125 GLL points are selected. 3 if 27 GLL points.
    :param ngll_y:    The number of GLL point in Y direction. As same as ngll_x.
    :param ngll_z:    The number of GLL point in Z direction. As same as ngll_x.
    :return:
            the interpolated SGT.
    '''

    n_step, n_dim, n_para = np.shape(sgt_arr_list[0])
    sgt_interp = np.zeros([n_step, n_dim, n_para])

    idx_sgt = 0
    for i in range(ngll_x):
        for j in range(ngll_y):
            for k in range(ngll_z):
                sgt_interp += h_xi_arr[i] * h_eta_arr[j] * h_gamma_arr[k] * sgt_arr_list[idx_sgt]
                idx_sgt += 1

    return sgt_interp
