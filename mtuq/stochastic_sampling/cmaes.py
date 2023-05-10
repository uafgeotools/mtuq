import numpy as np
from mtuq.util.cmaes import *
from mtuq.util.math import to_mij, to_rtp, to_rho
from mtuq.grid.force import to_force



# class CMA_ES(object):


class CMA_ES(object):

    def __init__(self, parameters_list, lmbda=None, data=None, GFclient=None, origin=None, callback_function=None): # Initialise with the parameters to be used in optimisation.

        # Initialize parameters-tied variables.
        self._parameters = parameters_list
        self.n = len(self._parameters)
        # Parameter setting
        if not lmbda == None:
            self.lmbda = lmbda
        else:
            self.lmbda = int((4 + np.floor(3*np.log(self.n)))*10) 
        self.xmean = np.asarray([[val.initial for val in self._parameters]]).T
        self.sigma = 2
        self.catalog_origin = origin
        self.counteval = 0
        if not callback_function == None:
            self.callback = callback_function
        elif len(parameters_list) == 6 or len(parameters_list) == 9:
            self.callback = to_mij
            self.mij_args = ['rho', 'v', 'w', 'kappa', 'sigma', 'h']
        elif len(parameters_list) == 3:
            self.callback = to_force

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

    def draw_mutants(self):
        bounds = [0,10]
        for i in range(self.lmbda):
            mutant = self.xmean + self.sigma * self.B @ (self.D * np.random.randn(self.n,1))
            self.mutants[:,i] = mutant.T
        for _i, param in enumerate(self.mutants):
            if not self._parameters[_i].repair == None:
                while array_in_bounds(self.mutants[_i], 0, 10) == False:
                    print('repairing '+self._parameters[_i].name+' with '+self._parameters[_i].repair+' method')
                    Repair(self._parameters[_i].repair, self.mutants[_i], self.xmean[_i])

    def mean_diff(self, new, old):
        diff = new-old
        for _i, param in enumerate(self._parameters):
            if param.repair == 'wrapping':
                angular_diff = linear_transform(new[_i], 0, 360)-linear_transform(old[_i], 0, 360)
                angular_diff = inverse_linear_transform((angular_diff+180)%360 - 180, 0, 360)
                diff[_i] = angular_diff
        return diff

    def circular_mean(self, id):
        param = self.mutants[id]
        a = linear_transform(param, 0, 360)-180
        mean = np.rad2deg(np.arctan2(np.sum(np.sin(np.deg2rad(a[range(int(self.mu))]))*self.weights.T), np.sum(np.cos(np.deg2rad(a[range(int(self.mu))]))*self.weights.T)))+180
        mean = inverse_linear_transform(mean, 0, 360)
        return mean

    # Evaluate the misfit for each mutant of the population.
    def eval_fitness(self, data, stations, db, process, misfit, wavelet):
        # Project each parameter in their respective physical domain, according to their `scaling` property
        self.transformed_mutants = np.zeros_like(self.mutants)
        for _i, param in enumerate(self._parameters):
            print(param.scaling)
            if param.scaling == 'linear':
                self.transformed_mutants[_i] = linear_transform(self.mutants[_i], param.lower_bound, param.upper_bound)
            elif param.scaling == 'log':
                self.transformed_mutants[_i] = logarithmic_transform(self.mutants[_i], param.lower_bound, param.upper_bound)
            else:
                raise ValueError("Unrecognized scaling, must be linear or log")
            # Apply optional projection operator to each parameter
            if not param.projection is None:
                self.transformed_mutants[_i] = np.asarray(list(map(param.projection, self.transformed_mutants[_i])))

        self.sources = np.ascontiguousarray(self.callback(*self.transformed_mutants[0:6]))
        print('creating new origins list')
        self.create_origins()
        for X in self.origins:
            print(X)
        self.local_greens = db.get_greens_tensors(stations, self.origins)
        self.local_greens.convolve(wavelet)
        self.local_greens = self.local_greens.map(process)
        self.misfit_val = [misfit(data, self.local_greens.select(origin), np.array([self.sources[_i]])) for _i, origin in enumerate(self.origins)]
        self.misfit_val = np.asarray(self.misfit_val).T[0]

        self.counteval += self.lmbda
        return self.misfit_val

    def fitness_sort(self, misfit):
        # Sort by fitness
        self.mutants = self.mutants[:,np.argsort(misfit)[0]]
        self.transformed_mutants = self.transformed_mutants[:,np.argsort(misfit)[0]]

    def update_mean(self):
        # Update the mean
        self.xold = self.xmean
        self.xmean = np.array([np.sum(self.mutants[:,range(int(self.mu))]*self.weights.T, axis=1)]).T
        for _i, param in enumerate(self._parameters):
            if param.repair == 'wrapping':
                print('computing wrapped mean for parameter:', param.name)
                self.xmean[_i] = self.circular_mean(_i)


    def update_step_size(self):
        # Step size control
        self.ps = (1-self.cs)*self.ps + np.sqrt(self.cs*(2-self.cs)*self.mueff) * self.invsqrtC @ (self.mean_diff(self.xmean, self.xold) / self.sigma)

    def update_covariance(self):
        # Covariance matrix adaptation
        if np.linalg.norm(self.ps)/np.sqrt(1-(1-self.cs)**(2*self.counteval/self.lmbda))/self.chin < 1.4 + 2/(self.n+1):
            self.hsig = 1
        if np.linalg.norm(self.ps)/np.sqrt(1-(1-self.cs)**(2*self.counteval/self.lmbda))/self.chin >= 1.4 + 2/(self.n+1):
            self.hsig = 0
        self.pc = (1-self.cc) * self.pc + self.hsig * np.sqrt(self.cc*(2-self.cc)*self.mueff) * self.mean_diff(self.xmean, self.xold) / self.sigma

        artmp = (1/self.sigma) * self.mean_diff(self.mutants[:,0:int(self.mu)], self.xold)
        self.C = (1-self.c1-self.cmu) * self.C + self.c1 * (self.pc@self.pc.T + (1-self.hsig) * self.cc*(2-self.cc) * self.C) + self.cmu * artmp @ np.diag(self.weights.T[0]) @ artmp.T
        # Adapt step size
        self.sigma = self.sigma * np.exp((self.cs/self.damps)*(np.linalg.norm(self.ps)/self.chin - 1))

        if self.counteval - self.eigeneval > self.lmbda/(self.c1+self.cmu)/self.n/10:
            self.eigeneval = self.counteval
            self.C = np.triu(self.C) + np.triu(self.C,1).T
            self.D,self.B = np.linalg.eig(self.C)
            self.D = np.array([self.D]).T
            self.D = np.sqrt(self.D)
            self.invsqrtC = self.B @ np.diag(self.D[:,0]**-1) @ self.B.T

    def create_origins(self):
        self.catalog_origin

        depth = self.transformed_mutants[6]
        latitude = self.transformed_mutants[7]
        longitude = self.transformed_mutants[8]

        self.origins = []
        for i in range(len(depth)):
            self.origins += [self.catalog_origin.copy()]
            setattr(self.origins[-1], 'depth_in_m', depth[i])
            setattr(self.origins[-1], 'latitude', latitude[i])
            setattr(self.origins[-1], 'longitude', longitude[i])

    def return_candidate_solution(self, id=None):
        if not id == None:
            return_x = np.array([self.mutants[:,id]]).T
        else:
            return_x = self.xmean
        self.transformed_mean = np.zeros_like(return_x)
        for _i, param in enumerate(self._parameters):
            print(param.scaling)
            if param.scaling == 'linear':
                self.transformed_mean[_i] = linear_transform(return_x[_i], param.lower_bound, param.upper_bound)
            elif param.scaling == 'log':
                self.transformed_mean[_i] = logarithmic_transform(return_x[_i], param.lower_bound, param.upper_bound)
            else:
                raise ValueError("Unrecognized scaling, must be linear or log")
            # Apply optional projection operator to each parameter
            if not param.projection is None:
                self.transformed_mean[_i] = np.asarray(list(map(param.projection, self.transformed_mean[_i])))

        self.catalog_origin
        depth = self.transformed_mean[6]
        latitude = self.transformed_mean[7]
        longitude = self.transformed_mean[8]
        self.final_origin = []
        for i in range(len(depth)):
            self.final_origin += [self.catalog_origin.copy()]
            setattr(self.final_origin[-1], 'depth_in_m', depth[i])
            setattr(self.final_origin[-1], 'latitude', latitude[i])
            setattr(self.final_origin[-1], 'longitude', longitude[i])

        return(self.transformed_mean, self.final_origin)
