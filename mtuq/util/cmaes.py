
import numpy as np
import pandas as pd

def linear_transform(i, a, b):
    """ Linear map suggested by N. Hansen for appropriate parameter scaling/variable encoding in CMA-ES

    Linear map from [0;10] to [a,b]

    source:
    (https://cma-es.github.io/cmaes_sourcecode_page.html)
    """
    transformed = a + (b-a) * i / 10
    return transformed

def inverse_linear_transform(transformed, a, b):
    """ Inverse linear mapping to reproject the variable in the [0; 10] range, from its original transformation bounds.

    """
    i = (10*(transformed-a))/(b-a)
    return i

def logarithmic_transform(i, a, b):
    """ Logarithmic mapping suggested  by N. Hansen. particularly adapted to define Magnitude ranges of [1e^(n),1e^(n+3)].

    Logarithmic map from [0,10] to [a,b], with `a` and `b` typically spaced by 3 to 4 orders of magnitudes.

    example usage:
    x = np.arange(0,11)
    projected_data = logarithmic_transform(x, 1e1, 1e4)

    source:
    (https://cma-es.github.io/cmaes_sourcecode_page.html)
    """
    d=np.log10(b)-np.log10(a)
    transformed = 10**(np.log10(a)) * 10**(d*i/10)
    return transformed

def in_bounds(value, a=0, b=10):
    return value >= a and value <= b

def array_in_bounds(array, a=0, b=10):
    """
    Check if all elements of an array are in bounds.
    :param array: The array to check.
    :param a: The lower bound.
    :param b: The upper bound.
    :return: True if all elements of the array are in bounds, False otherwise.
    """
    for i in range(len(array)):
        if not in_bounds(array[i], a, b):
            return False
    return True

class Repair:
    """
    Repair class to define all the boundary handling constraint method implemented in R. Biedrzycki 2019, https://doi.org/10.1016/j.swevo.2019.100627.

    These methods are invoqued whenever a CMA-ES mutant infringes a boundary.

    """
    def __init__(self, method, data_array, mean, lower_bound=0, upper_bound=10):
        self.method = method
        self.data_array = data_array
        self.mean = mean
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # Out of boundary violators mask. l_oob for lower_out-of-boundary and u_oob for upper_out-of-boundary.
        self.l_oob = self.data_array < self.lower_bound
        self.u_oob = self.data_array > self.upper_bound


        if self.method == 'reinitialize':
            # Call the reinitialize function
            self.reinitialize()
        elif self.method == 'projection':
            # Call the projection function
            self.projection()
        elif self.method == 'reflection':
            # Call the reflection function
            self.reflection()
        elif self.method == 'wrapping':
            # Call the wrapping function
            self.wrapping()
        elif self.method == 'transformation':
            # Call the transformation function
            self.transformation()
        elif self.method == 'projection_to_midpoint':
            # Call the projection_to_midpoint function
            self.projection_to_midpoint()
        elif self.method == 'rand_based':
            # Call the rand_based function
            self.rand_based()
        elif self.method == 'midpoint_base':
            # Call the rebound function
            self.midpoint_base()
        elif self.method == 'midpoint_target':
            # Call the rebound function
            self.midpoint_target()
        else:
            print('Repair method not recognized')

    def reinitialize(self):
        """
        Redraw all out of bounds values from a uniform distribution [0,10].
        """
        self.data_array[self.l_oob] = np.random.uniform(self.lower_bound,self.upper_bound, len(self.data_array[self.l_oob]))
        self.data_array[self.u_oob] = np.random.uniform(self.lower_bound,self.upper_bound, len(self.data_array[self.u_oob]))

    def projection(self):
        """
        Project all out of bounds values to the violated bounds.
        """
        self.data_array[self.l_oob] = self.lower_bound
        self.data_array[self.u_oob] = self.upper_bound

    def reflection(self):
        """
        The infeasible coordinate value of the solution is reflected back from the boundary by the amount of constraint violation.
        This method may be call several times if the points are out of bound for more than the length of the [0,10] domain.
        """
        self.data_array[self.l_oob] = 2*self.lower_bound - self.data_array[self.l_oob]
        self.data_array[self.u_oob] = 2*self.upper_bound - self.data_array[self.u_oob]

    def wrapping(self):
        """
        The infeasible coordinate value is shifted by the feasible interval.
        """
        self.data_array[self.l_oob] = self.data_array[self.l_oob] + self.upper_bound - self.lower_bound
        self.data_array[self.u_oob] = self.data_array[self.u_oob] - self.upper_bound + self.lower_bound

    def transformation(self):
        """
        Adaptive transformation based on 90%-tile -
        Nonlinear transform defined by R. Biedrzycki 2019, https://doi.org/10.1016/j.swevo.2019.100627
        """
        al = np.min([(self.upper_bound - self.lower_bound)/2,(1+np.abs(self.lower_bound-2))/20])
        au = np.min([(self.upper_bound - self.lower_bound)/2,(1+np.abs(self.upper_bound))/20])
        # Create the masks for the values out of self.lower_bound - al and self.upper_bound + au bounds
        mask_1 = self.data_array > (self.upper_bound + au)
        mask_2 = self.data_array < (self.lower_bound - al)
        # Reflect out of bounds values according to self.lower_bound - al and self.upper_bound + au.
        self.data_array[mask_1] = (2*self.upper_bound + au) - self.data_array[mask_1]
        self.data_array[mask_2] = 2*self.lower_bound - al - self.data_array[mask_2]

        # Create masks for the nonlinear transformation.
        mask_3 = (self.data_array >= (self.lower_bound + al)) & (self.data_array <= (self.upper_bound - au))
        mask_4 = (self.data_array >= (self.lower_bound - al)) & (self.data_array < (self.lower_bound + al))
        mask_5 = (self.data_array > (self.upper_bound - au)) & (self.data_array <= (self.upper_bound + au))

        # Note that reflected data are transformed according to the same principle which results in a periodic transformation
        self.data_array[mask_3] = self.data_array[mask_3]
        self.data_array[mask_4] = self.lower_bound + (self.data_array[mask_4] - (self.lower_bound-al))**2/(4*al)
        self.data_array[mask_5] = self.upper_bound - (self.data_array[mask_5] - (self.upper_bound-au))**2/(4*al)


    def projection_to_midpoint(self):
        """
        Project the particles onto the domain using the midpoint method (also reffered to as the Scaled Mutant)
        Get the farthest out-of-bounds mutant
        """
        largest_outlier = self.data_array[np.argmax(np.sqrt((self.data_array-(self.lower_bound+self.upper_bound)/2)**2))]

        alpha = np.abs(((self.lower_bound+self.upper_bound)/2)/(largest_outlier - (self.lower_bound+self.upper_bound)/2))
        self.data_array[:] = ((1-alpha)*5 + alpha*self.data_array[:])

    def rand_based(self):
        """
        Redraw the out of bound mutants between the base vector (the CMA_ES.xmean used in draw_muants()) and the violated boundary.
        The base vector is the mean of the population.
        """
        self.data_array[self.l_oob] = np.random.uniform(self.mean, self.lower_bound, len(self.data_array[self.l_oob]))
        self.data_array[self.u_oob] = np.random.uniform(self.mean, self.upper_bound, len(self.data_array[self.u_oob]))

    def midpoint_base(self):
        """
        The infeasible coordinate value is replaced by the midpoint between the base value and the violated boundary.
        """
        self.data_array[self.l_oob] = (self.mean + self.lower_bound)/2
        self.data_array[self.u_oob] = (self.mean + self.upper_bound)/2

    def midpoint_target(self):
        """
        The average of the target individual and the violated constraint replaces the infeasible coordinate value.
        """
        target = 5
        self.data_array[self.l_oob] = (target + self.lower_bound)/2
        self.data_array[self.u_oob] = (target + self.upper_bound)/2
