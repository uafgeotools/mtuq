import numpy as np
import pandas as pd
from mtuq.util.cmaes import *
from mtuq.util.math import to_mij, to_rtp
import mpi4py.MPI as MPI
from mtuq.greens_tensor import GreensTensor
from mtuq import MTUQDataFrame
from mtuq.grid.moment_tensor import UnstructuredGrid
from mtuq.grid.force import UnstructuredGrid
from mtuq.graphics import plot_data_greens2, plot_data_greens1
from mtuq.io.clients.AxiSEM_NetCDF import Client as AxiSEM_Client
from mtuq.greens_tensor.base import GreensTensorList
from mtuq.dataset import Dataset
from mtuq.event import MomentTensor, Force
from mtuq.graphics.uq._matplotlib import _hammer_projection, _generate_lune
from mtuq.util.math import to_gamma, to_delta
from mtuq.graphics import plot_combined
from mtuq.misfit import Misfit, PolarityMisfit
from mtuq.misfit.waveform import calculate_norm_data 
from mtuq.process_data import ProcessData
# class CMA_ES(object):


class CMA_ES(object):

    def __init__(self, parameters_list, lmbda=None, data=None, GFclient=None, origin=None, callback_function=None, event_id=''):
        '''
        parallel_CMA_ES class

        CMA-ES class for moment tensor and force inversion. The class accept a list of `CMAESParameters` objects containing the options and tunning of each of the inverted parameters. CMA-ES will be carried automatically based of the content of the `CMAESParameters` list.

        .. rubric :: Usage

        The inversion is carried in a two step procedure

        .. code::

        cma_es = parallel_CMA_ES(**parameters)
        cma_es.solve(data, process, misfit, stations, db, wavelet, iterations=10)

        .. note ::
        In the first step, the user supplies parameters such as the number of mutants, the list of inverted parameters, the catalog origin, etc. (see below for detailed argument descriptions).

        In the second step, the user supplies data, data process, misfit type, stations list, an Axisem Green's function database, a source wavelet and a number of iterations on which to carry the CMA-ES inversion (number of generations).

        .. rubric:: Parameters

        '''

        # Generate Class based MPI communicator
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()


        # Initialize parameters-tied variables.
        # Variables are initially shared across all processes, but most of the important computations will be carried on the root process (self.rank == 0) only.
        self.event_id = event_id
        self.iteration = 0
        self._parameters = parameters_list
        self._parameters_names = [parameter.name for parameter in parameters_list]
        self.n = len(self._parameters)
        self.xmean = np.asarray([[val.initial for val in self._parameters]]).T
        self.sigma = 0.5 # Default initial gaussian variance for all parameters.
        self.catalog_origin = origin
        self.counteval = 0
        self._greens_tensors_cache = {} # Minor optimization for `db` mode with fixed origin. Will store the Green's tensors for each ProcessData() used in eval_fitness(). While it is recommanded to use the `greens` mode for fixed origin, this will make the computation faster in case `db` mode is used instead.

        if self.rank == 0:
            print('Initialising CMA-ES inversion for event %s' % self.event_id)

        if not callback_function == None:
            self.callback = callback_function
        elif 'Mw' in self._parameters_names or 'kappa' in self._parameters_names:
            self.callback = to_mij
            self.mij_args = ['rho', 'v', 'w', 'kappa', 'sigma', 'h']
            self.mode = 'mt'
            if not 'w' in self._parameters_names and 'v' in self._parameters_names:
                self.mode = 'mt_dev'
                self.mij_args = ['rho', 'w', 'kappa', 'sigma', 'h']
            elif not 'v' in self._parameters_names and not 'w' in self._parameters_names:
                self.mode = 'mt_dc'
                self.mij_args = ['rho', 'kappa', 'sigma', 'h']
        elif 'F0' in self._parameters_names:
            self.callback = to_rtp
            self.mode = 'force'
        
        self.fig = None
        self.ax = None

        # Main user input: lmbda is the number of mutants. If no lambda is given, it will determine the number of mutants based on the number of parameters.
        if lmbda == None:
            self.lmbda = int(4 + np.floor(3*np.log(len(self._parameters))))
        else:
            self.lmbda = lmbda

        self.mu = np.floor(self.lmbda/2)
        a = 1 # Original author uses 1/2 in tutorial and 1 in publication
        self.weights = np.array([np.log(self.mu+a) - np.log(np.arange(1, self.mu+1))]).T
        self.weights /= sum(self.weights)
        self.mueff = sum(self.weights)**2/sum(self.weights**2)

        # Step-size control
        self.cs = (self.mueff + 2) / (self.n + self.mueff + 5)
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1)/(self.n + 1)) - 1) + self.cs

        # Covariance matrix adaptation
        self.cc = (4 + self.mueff / self.n)/(self.n + 4 + 2 * self.mueff / self.n)
        self.acov = 2
        self.c1 = self.acov / ((self.n + 1.3)**2 + self.mueff)
        self.cmu = min(1 - self.c1, self.acov * (self.mueff - 2 + 1 / self.mueff) / ((self.n + 2)**2 + self.acov*self.mueff/2))

        # Defining 'holder' variable for post processing and plotting.
        self._misfit_holder = np.zeros((int(self.lmbda),1))
        self.mutants_logger_list = pd.DataFrame()
        self.mean_logger_list = pd.DataFrame()

        # INITIALIZATION
        self.ps = np.zeros_like(self.xmean)
        self.pc = np.zeros_like(self.xmean)
        self.B = np.eye(self.n,self.n)
        self.D = np.ones((self.n, 1))
        self.C = self.B @ np.diag(self.D[:,0]**2) @ self.B.T
        self.invsqrtC = self.B @ np.diag(self.D[:,0]**-1) @ self.B.T
        self.eigeneval = 0
        self.chin = self.n**0.5 * (1 - 1 / ( 4 * self.n) + 1 / (21 * self.n**2))
        self.mutants = np.zeros((self.n, self.lmbda))

    # Where the CMA-ES Routine happens --------------------------------------------------------------
    def draw_mutants(self):
        '''
        draw_mutants method
        
        This function is responsible for drawing `self.lmbda` mutants from a Gaussian distribution on the root process. 
        It also applies the corresponding repair method for each of the parameters and then scatters the splitted list of mutants across all the computing processes.

        .. rubric:: Attributes

        self.lmbda (int):
            The number of mutants to be generated.
        self.xmean (array):
            The mean value of the distribution from which mutants are drawn.
        self.sigma (float):
            The standard deviation of the distribution.
        self.B (array):
            Eigenvectors of the covariance matrix (directions).
        self.D (array):
            Eigenvalues of the covariance matrix (amplitudes).
        self.n (int):
            Number of parameter to be inverted.
        self.mutants (array):
            The generated mutants. Each column corresponds to a mutant.
        self._parameters (list):
            A list of parameter objects, where each parameter has its own repair method.
        self.size (int):
            The number of processes.
        self.mutant_lists (list):
            A list of mutants divided amongst all processes.
        self.counteval (int):
            Counter for the total number of misfit evaluations.
        '''

        if self.rank == 0:
            # Hardcode the bounds of CMA-ES search. This forces the relative scaling between parameters.
            bounds = [0,10]

            # Randomly draw initial mutants from a Gaussian distribution
            for i in range(self.lmbda):
                mutant = self.xmean + self.sigma * self.B @ (self.D * np.random.randn(self.n,1))
                self.mutants[:,i] = mutant.T

            # Loop through all parameters to get their repair methods
            for _i, param in enumerate(self.mutants):
                if not self._parameters[_i].repair == None:
                    # If samples are out of the [0,10] range, apply repair method
                    while array_in_bounds(self.mutants[_i], bounds[0], bounds[1]) == False:
                        print('repairing '+self._parameters[_i].name+' with '+self._parameters[_i].repair+' method')
                        Repair(self._parameters[_i].repair, self.mutants[_i], self.xmean[_i])

            # Split the list of mutants evenly across all processes
            self.mutant_lists = np.array_split(self.mutants, self.size, axis=1)
        else:
            self.mutant_lists = None
        # Scatter the splited mutant_lists across processes.
        self.scattered_mutants = self.comm.scatter(self.mutant_lists, root=0)

        # Increase the counter of misfit evaluations (each mutant drawn now will be evaluated once)
        self.counteval += self.lmbda
    # Evaluate the misfit for each mutant of the population.
    def eval_fitness(self, data, stations, misfit, db_or_greens_list, process=None, wavelet=None, verbose=False):
        """
        eval_fitness method

        This method evaluates the misfit for each mutant of the population.

        .. rubric :: Usage

        The usage is as follows:

        .. code::

            if mode == 'db':
                eval_fitness(data, stations, misfit, db, process, wavelet)
            elif mode == 'greens':
                eval_fitness(data, stations, misfit, greens, process = None, wavelet = None)

        .. note ::
        The ordering of the CMA_ES parameters should follow the ordering of the input variables of the callback function, but this is dealt with internally if using the initialize_mt() and initialize_force() functions.

        .. rubric:: Parameters

        data (mtuq.Dataset): the data to fit (body waves, surface waves).
        stations (list): the list of stations.
        misfit (mtuq.WaveformMisfit): the associated mtuq.Misfit object.
        db (mtuq.AxiSEM_Client or mtuq.GreensTensorList): Preprocessed Greens functions or local database (for origin search).
        process (mtuq.ProcessData, optional): the processing function to apply to the Greens functions.
        wavelet (mtuq.wavelet, optional): the wavelet to convolve with the Greens functions.
        verbose (bool, optional): whether to print debug information.

        .. rubric:: Returns

        The misfit values for each mutant of the population.

        """
        # Check if the input parameters are valid
        self._check_greens_input_combination(db_or_greens_list, process, wavelet)

        # Use consistent coding style and formatting
        mode = 'db' if isinstance(db_or_greens_list, AxiSEM_Client) else 'greens'

        self._transform_mutants()

        self._generate_sources()

        if mode == 'db':
            # Check if latitude longitude AND depth are absent from the parameters list
            if not any(x in self._parameters_names for x in ['depth', 'latitude', 'longitude']):
                # If so, use the catalog origin, and make one copy per mutant to match the number of mutants.
                if self.rank == 0 and verbose == True:
                    print('using catalog origin')
                self.origins = [self.catalog_origin]

                key = self._get_greens_tensors_key(process)

                # Only rank 0 fetches the data from the database
                if self.rank == 0:
                    if key not in self._greens_tensors_cache:
                        self._greens_tensors_cache[key] = db_or_greens_list.get_greens_tensors(stations, self.origins)
                        self._greens_tensors_cache[key].convolve(wavelet)
                        self._greens_tensors_cache[key] = self._greens_tensors_cache[key].map(process)
                else:
                    self._greens_tensors_cache[key] = None

                # Rank 0 broadcasts the data to the others
                self.local_greens = self.comm.bcast(self._greens_tensors_cache[key], root=0)

                
                self.local_misfit_val = misfit(data, self.local_greens, self.sources)
                self.local_misfit_val = np.asarray(self.local_misfit_val).T
                if verbose == True:
                    print("local misfit is :", self.local_misfit_val) # - DEBUG PRINT

                # Gather local misfit values
                self.misfit_val = self.comm.gather(self.local_misfit_val.T, root=0)
                # Broadcast the gathered values and concatenate to return across processes.
                self.misfit_val = self.comm.bcast(self.misfit_val, root=0)
                self.misfit_val = np.asarray(np.concatenate(self.misfit_val)).T
                self._misfit_holder += self.misfit_val.T
                return self.misfit_val.T
            # If one of the three is present, create a list of origins (one for each mutant), and load the corresponding local greens functions.
            else:
                if self.rank == 0 and verbose == True:
                    print('creating new origins list')
                self.create_origins()
                if verbose == True:
                    for X in self.origins:
                        print(X)
                # Load, convolve and process local greens function
                start_time = MPI.Wtime()
                self.local_greens = db_or_greens_list.get_greens_tensors(stations, self.origins)
                end_time = MPI.Wtime()
                if self.rank == 0:
                    print("Fetching greens tensor: " + str(end_time-start_time))
                start_time = MPI.Wtime()
                self.local_greens.convolve(wavelet)
                end_time = MPI.Wtime()
                if self.rank == 0:
                    print("Convolution: " + str(end_time-start_time))
                start_time = MPI.Wtime()
                self.local_greens = self.local_greens.map(process)
                end_time = MPI.Wtime()
                if self.rank == 0:
                    print("Processing: " + str(end_time-start_time))
                # DEBUG PRINT to check what is happening on each process: print the number of greens functions loaded on each process
                if verbose == True:
                    print("Number of greens functions loaded on process", self.rank, ":", len(self.local_greens))


                # Compute misfit
                start_time = MPI.Wtime()
                self.local_misfit_val = [misfit(data, self.local_greens.select(origin), np.array([self.sources[_i]])) for _i, origin in enumerate(self.origins)]
                self.local_misfit_val = np.asarray(self.local_misfit_val).T[0]
                end_time = MPI.Wtime()

                if verbose == True:
                    print("local misfit is :", self.local_misfit_val) # - DEBUG PRINT

                if self.rank == 0:
                    print("Misfit: " + str(end_time-start_time))
                # Gather local misfit values
                self.misfit_val = self.comm.gather(self.local_misfit_val.T, root=0)
                # Broadcast the gathered values and concatenate to return across processes.
                self.misfit_val = self.comm.bcast(self.misfit_val, root=0)
                self.misfit_val = np.asarray(np.concatenate(self.misfit_val))
                self._misfit_holder += self.misfit_val
                return self.misfit_val
            
        elif mode == 'greens':
            # Check if latitude longitude AND depth are absent from the parameters list
            if not any(x in self._parameters_names for x in ['depth', 'latitude', 'longitude']):
                # If so, use the catalog origin, and make one copy per mutant to match the number of mutants.
                if self.rank == 0 and verbose == True:
                    print('using catalog origin')
                self.origins = [self.catalog_origin]
                self.local_greens = db_or_greens_list
                self.local_misfit_val = misfit(data, self.local_greens, self.sources)
                self.local_misfit_val = np.asarray(self.local_misfit_val).T
                if verbose == True:
                    print("local misfit is :", self.local_misfit_val)

                # Gather local misfit values
                self.misfit_val = self.comm.gather(self.local_misfit_val.T, root=0)
                # Broadcast the gathered values and concatenate to return across processes.
                self.misfit_val = self.comm.bcast(self.misfit_val, root=0)
                self.misfit_val = np.asarray(np.concatenate(self.misfit_val)).T
                self._misfit_holder += self.misfit_val.T
                return self.misfit_val.T
            # If one of the three is present, issue a warning and break.
            else:
                if self.rank == 0:
                    print('WARNING: Greens mode is not compatible with latitude, longitude or depth parameters. Consider using a local Axisem database instead.')
                return None
    # Gather the mutants from each process and concatenate them into a single array.
    def gather_mutants(self, verbose=True):
        '''
        gather_mutants method

        This function gathers mutants from all processes into the root process. It also uses the datalogger to construct the mutants_logger_list.

        .. rubric :: Usage

        The method is used as follows:

        .. code::

            gather_mutants(verbose=False)

        .. rubric:: Parameters

        verbose (bool):
            If set to True, prints the concatenated mutants, their shapes, and types. Default is False.

        .. rubric:: Attributes

        self.mutants (array):
            The gathered and concatenated mutants. This attribute is set to None for non-root processes after gathering.
        self.transformed_mutants (array):
            The gathered and concatenated transformed mutants. This attribute is set to None for non-root processes after gathering.
        self.mutants_logger_list (list):
            The list to which the datalogger is appended.
        '''

        # Printing the mutants on each process, their shapes and types for debugging purposes
        if verbose == True:
            print(self.scattered_mutants, '\n', 'shape is', np.shape(self.scattered_mutants), '\n', 'type is', type(self.scattered_mutants))


        self.mutants = self.comm.gather(self.scattered_mutants, root=0)
        if self.rank == 0:
            self.mutants = np.concatenate(self.mutants, axis=1)
            if verbose == True:
                print(self.mutants, '\n', 'shape is', np.shape(self.mutants), '\n', 'type is', type(self.mutants)) # - DEBUG PRINT
        else:
            self.mutants = None

        self.transformed_mutants = self.comm.gather(self.transformed_mutants, root=0)
        if self.rank == 0:
            self.transformed_mutants = np.concatenate(self.transformed_mutants, axis=1)
            if verbose == True:
                print(self.transformed_mutants, '\n', 'shape is', np.shape(self.transformed_mutants), '\n', 'type is', type(self.transformed_mutants)) # - DEBUG PRINT
        else:
            self.transformed_mutants = None

        self.mutants_logger_list = self.mutants_logger_list.append(
                            self._datalogger(mean=False))
    # Sort the mutants by fitness
    def fitness_sort(self, misfit):
        """
        fitness_sort method

        This function sorts the mutants by fitness, and updates the misfit_holder.

        .. rubric :: Usage

        The method is used as follows:

        .. code::

            fitness_sort(misfit)

        .. rubric:: Parameters

        misfit (array):

            The misfit array to sort the mutants by. Can be the sum of body and surface wave misfits, or the misfit of a single wave type.

        .. rubric:: Attributes

        self.mutants (array):
            The sorted mutants.
        self.transformed_mutants (array):
            The sorted transformed mutants.
        self._misfit_holder (array):
            The updated misfit_holder. Reset to 0 after sorting.

        """
        
        if self.rank == 0:
            self.mutants = self.mutants[:,np.argsort(misfit.T)[0]]
            self.transformed_mutants = self.transformed_mutants[:,np.argsort(misfit.T)[0]]
        self._misfit_holder *= 0
    # Update step size
    def update_step_size(self):
        # Step size control
        if self.rank == 0:
            self.ps = (1-self.cs)*self.ps + np.sqrt(self.cs*(2-self.cs)*self.mueff) * self.invsqrtC @ (self.mean_diff(self.xmean, self.xold) / self.sigma)
    # Update covariance matrix
    def update_covariance(self):
        # Covariance matrix adaptation
        if self.rank == 0:
            ps_norm = np.linalg.norm(self.ps)
            condition = ps_norm / np.sqrt(1 - (1 - self.cs) ** (2 * (self.counteval // self.lmbda + 1))) / self.chin
            threshold = 1.4 + 2 / (self.n + 1)

            if condition < threshold:
                self.hsig = 1
            else:
                self.hsig = 0

            self.dhsig = (1 - self.hsig)*self.cc*(2-self.cc)

            self.pc = (1 - self.cc) * self.pc + self.hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * self.mean_diff(self.xmean, self.xold) / self.sigma

            artmp = (1/self.sigma) * self.mean_diff(self.mutants[:,0:int(self.mu)], self.xold)
            # Old version - from the pureCMA Matlab implementation on Wikipedia
            # self.C = (1-self.c1-self.cmu) * self.C + self.c1 * (self.pc@self.pc.T + (1-self.hsig) * self.cc*(2-self.cc) * self.C) + self.cmu * artmp @ np.diag(self.weights.T[0]) @ artmp.T
            # Old version - from CMA-ES tutorial by Hansen et al. (2016) - https://arxiv.org/pdf/1604.00772.pdf
            # self.C = (1 + self.c1*self.dhsig - self.c1 - self.cmu*np.sum(self.weights)) * self.C + self.c1 * self.pc@self.pc.T + self.cmu * artmp @ np.diag(self.weights.T[0]) @ artmp.T

            # New version - from the purecma python implementation on GitHub - September, 2017, version 3.0.0
            # https://github.com/CMA-ES/pycma/blob/development/cma/purecma.py
            self.C *= 1 + self.c1*self.dhsig - self.c1 - self.cmu * sum(self.weights) # Discount factor
            self.C += self.c1 * self.pc @ self.pc.T # Rank one update (pc.pc^T is a matrix of rank 1) 
            self.C += self.cmu * artmp @ np.diag(self.weights.T[0]) @ artmp.T # Rank mu update, supported by the mu best individuals

            # Adapt step size
            # We do sigma_i+1 = sigma_i * exp((cs/damps)*(||ps||/E[N(0,I)]) - 1) only now as artmp needs sigma_i
            self.sigma = self.sigma * np.exp((self.cs/self.damps)*(np.linalg.norm(self.ps)/self.chin - 1))

            if self.counteval - self.eigeneval > self.lmbda/(self.c1+self.cmu)/self.n/10:
                self.eigeneval = self.counteval
                self.C = np.triu(self.C) + np.triu(self.C,1).T
                self.D,self.B = np.linalg.eig(self.C)
                self.D = np.array([self.D]).T
                self.D = np.sqrt(self.D)
                self.invsqrtC = self.B @ np.diag(self.D[:,0]**-1) @ self.B.T
        
        self.iteration = self.counteval//self.lmbda
    # Update mean
    def update_mean(self):
        # Update the mean
        if self.rank == 0:
            self.xold = self.xmean.copy()
            self.xmean = np.dot(self.mutants[:, 0:len(self.weights)], self.weights)
            for _i, param in enumerate(self._parameters):
                if param.repair == 'wrapping':
                    print('computing wrapped mean for parameter:', param.name)
                    self.xmean[_i] = self.circular_mean(_i)
            self.mean_logger_list=self.mean_logger_list.append(
                                self._datalogger(mean=True), ignore_index=True)
    # Utility functions --------------------------------------------------------------
    def circular_mean(self, id):
        '''
        Compute the circular mean on the "id"th parameter. Ciruclar mean allows to compute mean of the samples on a periodic space.
        '''
        param = self.mutants[id]
        a = linear_transform(param, 0, 360)-180
        mean = np.rad2deg(np.arctan2(np.sum(np.sin(np.deg2rad(a[range(int(self.mu))]))*self.weights.T), np.sum(np.cos(np.deg2rad(a[range(int(self.mu))]))*self.weights.T)))+180
        mean = inverse_linear_transform(mean, 0, 360)
        return mean

    def smallestAngle(self, targetAngles, currentAngles) -> np.ndarray:
        """
        smallestAngle method

        This function calculates the smallest angle (in degrees) between two given sets of angles. It computes the difference between the target and current angles, making sure the result stays within the range [0, 360). If the resulting difference is more than 180, it is adjusted to go in the shorter, negative direction.

        .. rubric :: Usage

        The method is used as follows:

        .. code::

            smallest_diff = smallestAngle(targetAngles, currentAngles)

        .. rubric:: Parameters

        targetAngles (np.ndarray):
            An array containing the target angles in degrees.
        currentAngles (np.ndarray):
            An array containing the current angles in degrees.

        .. rubric:: Returns

        diffs (np.ndarray):
            An array containing the smallest difference in degrees between the target and current angles.

        """

        # Subtract the angles, constraining the value to [0, 360)
        diffs = (targetAngles - currentAngles) % 360

        # If we are more than 180 we're taking the long way around.
        # Let's instead go in the shorter, negative direction
        diffs[diffs > 180] = -(360 - diffs[diffs > 180])
        return diffs

    def mean_diff(self, new, old):
        # Compute mean change, and apply circular difference for wrapped repair methods (implying periodic parameters)
        diff = new-old
        for _i, param in enumerate(self._parameters):
            if param.repair == 'wrapping':
                angular_diff = self.smallestAngle(linear_transform(new[_i], 0, 360), linear_transform(old[_i], 0, 360))
                angular_diff = inverse_linear_transform(angular_diff, 0, 360)
                diff[_i] = angular_diff
        return diff

    def create_origins(self):
        
        # Check which of the three origin modifiers are in the parameters
        if 'depth' in self._parameters_names:
            depth = self.transformed_mutants[self._parameters_names.index('depth')]
        else:
            depth = self.catalog_origin.depth_in_m
        if 'latitude' in self._parameters_names:
            latitude = self.transformed_mutants[self._parameters_names.index('latitude')]
        else:
            latitude = self.catalog_origin.latitude
        if 'longitude' in self._parameters_names:
            longitude = self.transformed_mutants[self._parameters_names.index('longitude')]
        else:
            longitude = self.catalog_origin.longitude
        
        self.origins = []
        for i in range(len(self.scattered_mutants[0])):
            self.origins += [self.catalog_origin.copy()]
            if 'depth' in self._parameters_names:
                setattr(self.origins[-1], 'depth_in_m', depth[i])
            if 'latitude' in self._parameters_names:
                setattr(self.origins[-1], 'latitude', latitude[i])
            if 'longitude' in self._parameters_names:
                setattr(self.origins[-1], 'longitude', longitude[i])

    def return_candidate_solution(self, id=None):
        # Only required on rank 0
        if self.rank == 0:
            if not id == None:
                return_x = np.array([self.mutants[:,id]]).T
            else:
                return_x = self.xmean
            self.transformed_mean = np.zeros_like(return_x)
            for _i, param in enumerate(self._parameters):
                # Print paramter scaling if verbose
                if param.scaling == 'linear':
                    self.transformed_mean[_i] = linear_transform(return_x[_i], param.lower_bound, param.upper_bound)
                elif param.scaling == 'log':
                    self.transformed_mean[_i] = logarithmic_transform(return_x[_i], param.lower_bound, param.upper_bound)
                else:
                    raise ValueError("Unrecognized scaling, must be linear or log")
                # Apply optional projection operator to each parameter
                if not param.projection is None:
                    self.transformed_mean[_i] = np.asarray(list(map(param.projection, self.transformed_mean[_i])))


            # Check which of the three origin modifiers are in the parameters
            if 'depth' in self._parameters_names:
                depth = self.transformed_mean[self._parameters_names.index('depth')]
            else:
                depth = self.catalog_origin.depth_in_m
            if 'latitude' in self._parameters_names:
                latitude = self.transformed_mean[self._parameters_names.index('latitude')]
            else:
                latitude = self.catalog_origin.latitude
            if 'longitude' in self._parameters_names:
                longitude = self.transformed_mean[self._parameters_names.index('longitude')]
            else:
                longitude = self.catalog_origin.longitude
            
            self.origins = []
            for i in range(len(self.transformed_mean[0])):
                self.origins += [self.catalog_origin.copy()]
                if 'depth' in self._parameters_names:
                    setattr(self.origins[-1], 'depth_in_m', depth[i])
                if 'latitude' in self._parameters_names:
                    setattr(self.origins[-1], 'latitude', latitude[i])
                if 'longitude' in self._parameters_names:
                    setattr(self.origins[-1], 'longitude', longitude[i])

            return(self.transformed_mean, self.origins)

    def _datalogger(self, mean=False):
        """
        _datalogger method

        This method saves in memory all of the CMA-ES mutants drawn and evaluated during the inversion. This allows quick access to the inversion records in order to plot the misfit. The data is stored within a pandas.DataFrame().

        Note
        ----------
        When mean=False, the datalogger stores the coordinates of each mutant (Mw, v, w, kappa, sigma,...) and misfit at the current iteration.

        When mean=True, the datalogger stores the coordinates of the mean mutant at the current iteration. The mean mutant's misfit is not evaluated, thus only its coordinates are returned.
            """
        if self.rank == 0:
            if mean==False:
                coordinates = self.transformed_mutants.T
                misfit_values = self._misfit_holder
                results = np.hstack((coordinates, misfit_values))
                columns_labels=self._parameters_names+["misfit"]

            if mean==True:
                self.transformed_mean = np.zeros_like(self.xmean)
                for _i, param in enumerate(self._parameters):
                    if param.scaling == 'linear':
                        self.transformed_mean[_i] = linear_transform(self.xmean[_i], param.lower_bound, param.upper_bound)
                    elif param.scaling == 'log':
                        self.transformed_mean[_i] = logarithmic_transform(self.xmean[_i], param.lower_bound, param.upper_bound)
                    else:
                        raise ValueError("Unrecognized scaling, must be linear or log")
                    # Apply optional projection operator to each parameter
                    if not param.projection is None:
                        self.transformed_mean[_i] = np.asarray(list(map(param.projection, self.transformed_mean[_i])))

                results = self.transformed_mean.T
                columns_labels=self._parameters_names

            da = pd.DataFrame(
            data=results,
            columns=columns_labels
            )
            return(MTUQDataFrame(da))

    def Solve(self, data_list, stations, misfit_list, process_list, db_or_greens_list, max_iter=100, wavelet=None, plot_interval=10, iter_count=0, **kwargs):
        """
        Solves for the best-fitting source model using the CMA-ES algorithm. This is the master method used in inversions. 

        This method iteratively draws mutants, evaluates their fitness based on misfits between synthetic and observed data, and updates the mean and covariance of the CMA-ES distribution. At specified intervals, it also plots mean waveforms and results for visualization.

        Parameters
        ----------
        data_list : list
            List of observed data sets. (e.g. [data_sw] or [data_bw, data_sw])
        stations : list
            List of stations (generally obtained from mtuq method data.get_stations())
        misfit_list : list
            List of mtuq misfit objects (e.g. [misfit_sw] or [misfit_bw, misfit_sw]).
        process_list : list
            List of mtuq ProcessData objects to apply to data (e.g. [process_sw] or [process_bw, process_sw]).
        db_or_greens_list : list or AxiSEM_Client object
            Either an AxiSEM database client or a mtuq GreensTensorList.
        max_iter : int, optional
            Maximum number of iterations to perform. Default is 100. A stoping criterion will be implemented in the future.
        wavelet : object, optional
            Wavelet for source time function. Default is None. Required when db_or_greens_list is an AxiSEM database client.
        plot_interval : int, optional
            Interval at which plots of mean waveforms and results should be generated. Default is 10.
        iter_count : int, optional
            Current iteration count, should be useful for resuming. Default is 0.
        src_type : str, optional
            Type of source model, one of ['full', 'deviatoric', 'dc']. Default is full.
        **kwargs
            Additional keyword arguments passed to eval_fitness method.

        Returns
        -------
        None

        Note
        ----
        This method is the wrapper that automate the execution of the CMA-ES algorithm. It is the default workflow for Moment tensor and Force inversion and should not work with a "custom" inversion (multiple-sub events, source time function, etc.). It interacts with the  `draw_mutants`, `eval_fitness`, `gather_mutants`, `fitness_sort`, `update_mean`, `update_step_size` and `update_covariance`. 
        """

        if self.rank == 0:
            # Check Solve inputs
            self._check_Solve_inputs(data_list, stations, misfit_list, process_list, db_or_greens_list, max_iter, wavelet, plot_interval, iter_count)

        for i in range(max_iter):
            if self.rank == 0:
                print('Iteration %d\n' % (i + iter_count))
            
            self.draw_mutants()

            misfits = []
            for j in range(len(data_list)):
                current_data = data_list[j]
                current_misfit = misfit_list[j]
                
                mode = 'db' if isinstance(db_or_greens_list, AxiSEM_Client) else 'greens'
                # get greens[j] or db depending on mode from db_or_greens_list
                greens = db_or_greens_list[j] if mode == 'greens' else None
                db = db_or_greens_list if mode == 'db' else None

                if mode == 'db':
                    misfit_values = self.eval_fitness(current_data, stations, current_misfit, db, process_list[j], wavelet, **kwargs)
                elif mode == 'greens':
                    misfit_values = self.eval_fitness(current_data, stations, current_misfit, greens,  **kwargs)

                norm = self._get_data_norm(current_data, current_misfit)
                        
                misfit_values = misfit_values/norm

                misfits.append(misfit_values)

                # # Print the local lists on each process:
                # print('Misfit on process %d: %s' % (self.rank, str(misfit_values)))

            self.gather_mutants()
            self.fitness_sort(sum(misfits)/len(misfits))
            self.update_mean()
            self.update_step_size()
            self.update_covariance()

            if i == 0 or i % plot_interval == 0 or i == max_iter - 1:
                self.plot_mean_waveforms(data_list, process_list, misfit_list, stations, db_or_greens_list)
                if self.mode in ['mt', 'mt-dc', 'mt-dev']:
                    if self.rank == 0:
                        print('Plotting results for iteration %d\n' % (i + iter_count))
                        result = self.mutants_logger_list
                        plot_combined('combined.png', result, colormap='viridis')

    def plot_mean_waveforms(self, data_list, process_list, misfit_list, stations, db_or_greens_list):
        """
        Plots the mean waveforms using the base mtuq waveform plots (mtuq.graphics.waveforms).

        Depending on the mode, different parameters are inserted into the mean solution (padding w or v with 0s for instance)
        If green's functions a provided directly, they are used as is. Otherwise, extrace green's function from Axisem database and preprocess them.
        Support only 1 or 2 waveform groups (body and surface waves, or surface waves only)

        Arguments
        ----------
            data_list: A list of data to be plotted (typically `data_bw` and `data_sw`).
            process_list: A list of processes for each data (typically `process_bw` and `process_sw`).
            misfit_list: A list of misfits for each data (typically `misfit_bw` and `misfit_sw`).
            stations: A list of stations.
            db_or_greens_list: Either an AxiSEM_Client instance or a list of GreensTensors (typically `greens_bw` and `greens_sw`).

        Raises
        ----------
            ValueError: If the mode is not 'mt', 'mt_dev', 'mt_dc', or 'force'.
        """

        if self.rank != 0:
            return  # Exit early if not rank 0

        mean_solution, final_origin = self.return_candidate_solution()

        # Solution grid will change depending on the mode (mt, mt_dev, mt_dc, or force)
        modes = {
            'mt': ('rho', 'v', 'w', 'kappa', 'sigma', 'h'),
            'mt_dev': ('rho', 'v', 'w', 'kappa', 'sigma', 'h'),
            'mt_dc': ('rho', 'v', 'w', 'kappa', 'sigma', 'h'),
            'force': ('F0', 'phi', 'h'),
        }

        if self.mode not in modes:
            raise ValueError("Invalid mode. Supported modes: 'mt', 'mt_dev', 'mt_dc', 'force'")

        mode_dimensions = modes[self.mode]

        # Pad mean_solution based on moment tensor mode (deviatoric or double couple)
        if self.mode == 'mt_dev':
            mean_solution = np.insert(mean_solution, 2, 0, axis=0)
        elif self.mode == 'mt_dc':
            mean_solution = np.insert(mean_solution, 1, 0, axis=0)
            mean_solution = np.insert(mean_solution, 2, 0, axis=0)

        solution_grid = UnstructuredGrid(dims=mode_dimensions, coords=mean_solution, callback=self.callback)

        final_origin = final_origin[0]
        if self.mode.startswith('mt'):
            best_source = MomentTensor(solution_grid.get(0))
        elif self.mode == 'force':
            best_source = Force(solution_grid.get(0))

        lune_dict = solution_grid.get_dict(0)

        # Assignments for brevity (might be removed later)
        data = data_list.copy()
        process = process_list.copy()
        misfit = misfit_list.copy()
        greens_or_db = db_or_greens_list.copy() if isinstance(db_or_greens_list, list) else db_or_greens_list

        # greens_or_db = db_or_greens_list

        mode = 'db' if isinstance(greens_or_db, AxiSEM_Client) else 'greens'
        if mode == 'db':
            _greens = greens_or_db.get_greens_tensors(stations, final_origin)
            greens = [None] * len(process_list)
            for i in range(len(process_list)):
                greens[i] = _greens.map(process_list[i])
                greens[i][0].tags[0] = 'model:ak135f_2s'
        elif mode == 'greens':
            greens = greens_or_db

        # Check for the occurences of mtuq.misfit.polarity.PolarityMisfit in misfit_list:
        # if present, remove the corresponding data, greens, process and misfit from the lists
        # Run backward to avoid index errors
        for i in range(len(misfit)-1, -1, -1):
            if isinstance(misfit[i], PolarityMisfit):
                del data[i]
                del process[i]
                del misfit[i]
                del greens[i]

        # Plot based on the number of ProcessData objects in the process_list
        if len(process) == 2:
            plot_data_greens2(self.event_id + 'FMT_waveforms_mean_' + str(self.iteration) + '.png',
                            data[0], data[1], greens[0], greens[1], process[0], process[1],
                            misfit[0], misfit[1], stations, final_origin, best_source, lune_dict)
        elif len(process) == 1:
            plot_data_greens1(self.event_id + 'FMT_waveforms_mean_' + str(self.iteration) + '.png',
                            data[0], greens[0], process[0], misfit[0], stations, final_origin, best_source, lune_dict)

    def _scatter_plot(self):
        """
        Generates a scatter plot of the mutants and the current mean solution
        
        Return: 
        Matplotlib figure object (also retrived by self.fig)
        """
        if self.rank == 0:
            if self.fig is None:  # Check if fig is None
                self.fig, self.ax = _generate_lune()

            # Define v as by values from self.mutants_logger_list if it exists, otherwise pad with values of zeroes
            m = np.asarray(self.mutants_logger_list['misfit'])

            if 'v' in self.mutants_logger_list:
                v = np.asarray(self.mutants_logger_list['v'])
            else:
                v = np.zeros_like(m)

            if 'w' in self.mutants_logger_list:
                w = np.asarray(self.mutants_logger_list['w'])
            else:
                w = np.zeros_like(m)
            
            # Handling the mean solution
            if self.mode == 'mt':
                V,W = self._datalogger(mean=True)['v'], self._datalogger(mean=True)['w']
            elif self.mode == 'mt_dev':
                V = self._datalogger(mean=True)['v']
                W = 0
            elif self.mode == 'mt_dc':
                V = W = 0

            # Projecting the mean solution onto the lune
            V, W = _hammer_projection(to_gamma(V), to_delta(W))
            self.ax.scatter(V, W, c='red', marker='x', zorder=10000000)
            # Projecting the mutants onto the lune
            v, w = _hammer_projection(to_gamma(v), to_delta(w))


            vmin, vmax = np.percentile(np.asarray(m), [0,90])

            self.ax.scatter(v, w, c=m, s=3, vmin=vmin, vmax=vmax, zorder=100)

            self.fig.canvas.draw()
            return self.fig

    def _transform_mutants(self):
        """
        Transforms local mutants on each process based on the parameters scaling and projection settings.

        For each parameter, depending on its scaling setting ('linear' or 'log'), 
        it applies a transformation to the corresponding elements of scattered_mutants.
        If a projection is specified, it applies this projection to the transformed values.

        Attributes
        ----------
            scattered_mutants: A 2D numpy array with the original mutant data. When MPI is used, correspond to the local mutants on each process.
            _parameters: A list of Parameter objects, each with attributes 'scaling', 'lower_bound', 'upper_bound', 
            and 'projection' specifying how to transform the corresponding scattered_mutants.

        Raises:
            ValueError: If an unrecognized scaling is provided.
        """        

        self.transformed_mutants = np.zeros_like(self.scattered_mutants)
        for i, param in enumerate(self._parameters):
            if param.scaling == 'linear':
                self.transformed_mutants[i] = linear_transform(self.scattered_mutants[i], param.lower_bound, param.upper_bound)
            elif param.scaling == 'log':
                self.transformed_mutants[i] = logarithmic_transform(self.scattered_mutants[i], param.lower_bound, param.upper_bound)
            else:
                raise ValueError("Unrecognized scaling, must be linear or log")
            if param.projection is not None:
                self.transformed_mutants[i] = np.asarray(list(map(param.projection, self.transformed_mutants[i])))

    def _generate_sources(self):
        """
        Generate sources by calling the callback function on transformed data according to the set mode.
        
        Depending on the mode, the method selects a subset of transformed mutants, possibly extending
        it with zero-filled columns at specific positions, and then passes the processed data to the
        callback function. The results are stored in a contiguous NumPy array in self.sources.

        Raises
        ----------
            ValueError: If an unsupported mode is provided.

        Attributes
        ----------
            mode: A string representing the mode of operation, which can be 'mt', 'mt_dev', 'mt_dc', or 'force'.
            transformed_mutants: A 2D numpy array that contains the transformed data to be processed.
            callback: A callable that is used to process the data.
        """

        # Mapping between modes and respective slices or insertion positions for processing.
        mode_to_indices = {
            'mt': (0, 6),        # For 'mt', a slice from the first 6 elements of transformed_mutants is used.
            'mt_dev': (0, 5, 2), # For 'mt_dev', a zero column is inserted at position 2 after slicing the first 5 elements.
            'mt_dc': (0, 4, 1, 2), # For 'mt_dc', zero columns are inserted at positions 1 and 2 after slicing the first 4 elements.
            'force': (0, 3),     # For 'force', a slice from the first 3 elements of transformed_mutants is used.
        }

        # Check the mode's validity. Raise an error if the mode is unsupported.
        if self.mode not in mode_to_indices:
            raise ValueError(f'Invalid mode: {self.mode}')

        # Get the slice or insertion positions based on the current mode.
        indices = mode_to_indices[self.mode]

        # If the mode is 'mt' or 'force', take a slice from transformed_mutants and pass it to the callback.
        if self.mode in ['mt', 'force']:
            self.sources = np.ascontiguousarray(self.callback(*self.transformed_mutants[indices[0]:indices[1]]))
        else:
            # For 'mt_dev' and 'mt_dc' modes, insert zeros at specific positions after slicing transformed_mutants.
            self.extended_mutants = self.transformed_mutants[indices[0]:indices[1]]
            for insertion_index in indices[2:]:
                self.extended_mutants = np.insert(self.extended_mutants, insertion_index, 0, axis=0)
            # Pass the processed data to the callback, and save the result as a contiguous array in self.sources.
            self.sources = np.ascontiguousarray(self.callback(*self.extended_mutants[0:6]))

    def _get_greens_tensors_key(self, process):
        """
        Get the body-wave or surface-wave key for the GreensTensors object from the ProcessData object.
        """
        return process.window_type

    def _check_greens_input_combination(self, db, process, wavelet):
        """
        Checks the validity of the given parameters.

        Raises a ValueError if the database object is not an AxiSEM_Client or GreensTensorList, 
        or if the process function and wavelet are not defined when the database object is an AxiSEM_Client.

        Arguments
        ----------
            db: The database object to check, expected to be an instance of either AxiSEM_Client or GreensTensorList.
            process: The process function to be used if the database is an AxiSEM_Client.
            wavelet: The wavelet to be used if the database is an AxiSEM_Client.

        Raises
        ----------
            ValueError: If the input combination of db, process, and wavelet is invalid.
        """

        if not isinstance(db, (AxiSEM_Client, GreensTensorList)):
            raise ValueError("database must be either an AxiSEM_Client object or a GreensTensorList object")
        if isinstance(db, AxiSEM_Client) and (process is None or wavelet is None):
            raise ValueError("process_function and wavelet must be specified if database is an AxiSEM_Client")

    def _check_Solve_inputs(self, data_list, stations, misfit_list, process_list, db_or_greens_list, max_iter=100, wavelet=None, plot_interval=10, iter_count=0, **kwargs):
        """
        Checks the validity of input arguments for the Solve method.
        
        Raises
        ------
        ValueError : If any of the inputs are invalid.
        """

        if not isinstance(data_list, list):
            if isinstance(data_list, Dataset):
                data_list = [data_list]
            else:
                raise ValueError("`data_list` should be a list of mtuq Dataset or an array containing polarities.")
        if not isinstance(stations, list):
            raise ValueError("`stations` should be a list of mtuq Station objects.")
        if not isinstance(misfit_list, list):
            if isinstance(misfit_list, PolarityMisfit) or isinstance(misfit_list, Misfit):
                misfit_list = [misfit_list]
            else:
                raise ValueError("`misfit_list` should be a list of mtuq Misfit objects.")
        if not isinstance(process_list, list):
            if isinstance(process_list, ProcessData):
                process_list = [process_list]
            else:
                raise ValueError("`process_list` should be a list of mtuq ProcessData objects.")
        if not isinstance(db_or_greens_list, list):
            if isinstance(db_or_greens_list, AxiSEM_Client) or isinstance(db_or_greens_list, GreensTensorList):
                db_or_greens_list = [db_or_greens_list]
            else:
                raise ValueError("`db_or_greens_list` should be a list of either mtuq AxiSEM_Client or GreensTensorList objects.")
        if not isinstance(max_iter, int):
            raise ValueError("`max_iter` should be an integer.")
        if any(isinstance(db, AxiSEM_Client) for db in db_or_greens_list) and wavelet is None:
            raise ValueError("wavelet must be specified if database is an AxiSEM_Client")
        if not isinstance(plot_interval, int):
            raise ValueError("`plot_interval` should be an integer.")
        if iter_count is not None and not isinstance(iter_count, int):
            raise ValueError("`iter_count` should be an integer or None.")            

    def _get_data_norm(self, data, misfit):
        """
        Compute the norm of the data using the calculate_norm_data function.

        Arguments
        ----------
            data: The evaluated processed data.
            misfit: The misfit object used to evaluate the data object

        """

        # If misfit type is Polarity misfit, use the sum of the absolute values of the data as number of used stations.
        if isinstance(misfit, PolarityMisfit):
            return np.sum(np.abs(data))
        # Else, use the calculate_norm_data function.
        else:
            if isinstance(misfit.time_shift_groups, str):
                components = list(misfit.time_shift_groups)
            elif isinstance(misfit.time_shift_groups, list):
                components = list("".join(misfit.time_shift_groups))
            
            return calculate_norm_data(data, misfit.norm, components)