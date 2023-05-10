import numpy as np
from mtuq.util.math import to_mij, to_rtp, to_rho
from mtuq.stochastic_sampling.cmaes_parallel import CMAESParameters


class CMAESSource:
    """ Source object for the parallel_CMA_ES class.

    This objects controls the source type accepted by the CMA-ES method.

    .. rubric:: Usage

    Encode each CMA-ES variable with a metadata-like structure. This class holds the name and search parameters of each variable, as well as the potential specific scaling, initial values, boundary handling repair methods and projection function.

    .. code::

        variable_name = CMAESParameters(**parameters)

    .. rubric:: Parameters

    ``name`` (`str`): name of the variable, for bookeeping purposes

    ``lower_bound`` (`float`): define the lower bound of the search space

    ``upper_bound`` (`float`): define the upper bound of the search space

    ``scaling`` (`str`):
        - ``'linear'``: linear scaling between lower_bound and upper_bound
        ..  Linear map from [0;10] to [a,b] ..

        - ``'log'``: logarithmic scaling between lower_bound and upper_bound
        ..  Logarithmic map from [0,10] to [a,b], with `a` and `b` typically defining 3 to 4 orders of magnitudes ..

    ``initial`` (`float`): Define the initial value within the 0 to 10 bound. If None, that value is initialised at random.

    ``repair`` (`float`): Define the boundary handling constraint method to deal with out of bound values (out of the 0 to 10 range). Default method is rand-based. See
        - ``'rand-based'``: Redraw the out of bound mutants between the base vector (the CMA_ES.xmean used in draw_muants()) and the violated boundary. The base vector is the mean of the population.

    ``projection`` (`method`): Assign a custom projection function to the state variable. The main use is to convert moment magnitude Mw to rho as in Tape and Tape 2015 uniform moment tensor parameterization.
        -``'mtuq.util.math.to_rho()'``


    """

    def __init__(self, name, lower_bound, upper_bound, scaling = 'linear', initial=None, repair='rand_based', projection=None, **kwargs):
        self.name = name # Name of the parameter, used for printing and saving to file names etc...
        if lower_bound > upper_bound:
            raise ValueError('Lower bound is larger than the Upper Bound')

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        if scaling not in ['linear', 'log']:
            raise ValueError('Scaling must be either linear or log')

        self.scaling=scaling # Define the scaling type to rescale the randomly drawn parameter in[0,10] to a value between self.lower_bound and self.upper_bound
        if initial == None:
            self.initial = np.random.uniform(0,10)
        elif initial < 0 or initial > 10:
            raise ValueError('Initial value is outside of the expected bounds')
        else:
            self.initial = initial # Initial guess for parameter values, used to seed CMA-ES optimisation algorithm
        self.repair = repair # Repair method used to redraw samples out of the [0,10] range.
        self.projection = projection

        if 'grid' in kwargs:
            self.grid = kwargs['grid'] # Grid of parameter values to use a non continuous optimization purposes. must be an array with initial and final value consistent with predefined lower_bound and upper_bound.
        else:
            self.grid = None


def initialise_mt(Mw_range=[4,5], depth_range=None, latitude_range=None, longitude_range=None):
    # Defining the inverted parameters to initialise the CMA-ES
    # The order expected by CMA-ES is set to 'Mw, v, w, kappa, sigma, h, depth, latitude, longitude', with depth, lat and lon being completely optional.

    Mw = CMAESParameters('Mw', Mw_range[0], Mw_range[1], scaling='log', repair='rand_based', projection=to_rho)
    v = CMAESParameters('v', -1/3, 1/3, scaling='linear', repair='rand_based')
    w = CMAESParameters('w', (-3/8*np.pi), (3/8*np.pi), scaling='linear', repair='rand_based')
    kappa = CMAESParameters('kappa', 0, 360, scaling='linear', repair='wrapping')
    sigma = CMAESParameters('sigma', -90, 90, scaling='linear', repair='rand_based')
    h = CMAESParameters('h', 0, 1, scaling='linear', repair='rand_based')

    parameters_list = [Mw, v, w, kappa, sigma, h]

    if depth_range:
        depth = CMAESParameters('depth', depth_range[0], depth_range[1])
        parameters_list += [depth]

    if latitude_range:
        latitude = CMAESParameters('latitude', latitude_range[0], latitude_range[1])
        parameters_list += [latitude]

    if longitude_range:
        longitude = CMAESParameters('longitude', longitude_range[0], longitude_range[1])
        parameters_list += [longitude]

    return(parameters_list)

def initialise_force(depth=None, latitude=None, longitude=None):
    raise NotImplementedError
