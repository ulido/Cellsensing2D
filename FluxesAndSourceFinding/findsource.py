import numpy as np
from numpy.linalg import norm
from scipy.optimize import fsolve
from tqdm.auto import tqdm

def G_H(x, y):
    '''Green's function for the 2D half-space
    
    `x` and `y` are Nx2 arrays.
    '''
    return -np.log(((x-y)**2).sum(axis=0))

def G_D(x, y):
    '''Green's function of the 2D reflecting disk.
    
    `x` and `y` are Nx2 arrays.
    '''
    yd = (y**2).sum(axis=0)
    return -np.log(((x-y)**2).sum(axis=0)*((x-y/yd)**2).sum(axis=0)/yd**0.5)

class Receptors:
    '''Describes N receptors.
    Can be subclassed to provide more efficient implementation for specific numbers of receptors.'''
    def __init__(self, x, eps, G, disk=False, guesses=None):
        '''Init receptors
          `x`: 2x2 arrays holding the locations of the receptors (need to be on a boundary)
          `eps`: Radius of receptors (typically 0.1)
          `G`: Function to calculate the Green's function of the domain
          `disk`: `True` if the domain is a reflecting disk (to exclude this)
        '''
        self.x = np.array(x)
        self.G = G
        self.eps = eps
        self.guesses = np.array(guesses)
        self.disk = disk
        
        self.initialise()

    def initialise(self):
        # Populate matrix to solve for receptor probabilities
        # Cf. equations (37-39) in https://doi.org/10.1016/j.jcp.2017.10.058
        N = self.x.shape[0]
        A = np.zeros((N+1, N+1))
        for i in range(N):
            for j in range(i+1, N):
                A[i, j] = A[j, i] = np.log(norm(self.x[i]-self.x[j])/self.eps)
        A[:N, N] = 1
        A[N, :N] = 1
        self.matrix = A

    def evaluate(self, x0):
        '''Get the probability of a particle being absorbed by the receptor windows when the source is at `x0`'''
        if self.disk:
            if (x0**2).sum() <= 1.0:
                return np.nan*np.ones(self.x.shape[0])
        
        return self._evaluate(x0)
    
    def _evaluate(self, x0):
        # Solve for probabilities
        # Cf. equations (37-39) in https://doi.org/10.1016/j.jcp.2017.10.058
        b = np.array([-self.G(xx, x0) for xx in self.x] + [1/np.pi])
        return np.pi*np.linalg.solve(self.matrix, b)[:-1]
        
    def find_point(self, guess, P):
        sol = fsolve(lambda x0: (self.evaluate(x0) - P)[:2], guess, full_output=True)
        return sol[2]==1, sol[0]

    def scan_P(self, P):
        if self.guesses is None:
            raise ValueError('Need a valid list of guesses to proceed.')
        x0 = np.nan*np.ones((2, P.shape[1]), dtype=float)
        for i in tqdm(range(P.shape[1])):
            for guess in self.guesses:
                converged, trial = self.find_point(guess, P[:, i])
                if converged:
                    x0[:, i] = trial
                    break
        return x0
    
class TwoReceptors(Receptors):
    '''Describes two receptors'''
    def initialise(self):
        if self.x.shape != (2, 2):
            raise ValueError('x needs to be a 2x2 array of receptor locations')
        # Precalculate relative receptor log distance
        self.D = np.log(norm(self.x[0]-self.x[1])/self.eps)
        
    def _evaluate(self, x0):
        P2 = 1/2 + np.pi/2 * (self.G(self.x[1], x0) - self.G(self.x[0], x0))/self.D
        return np.array([1-P2, P2])

class ThreeReceptors(Receptors):
    '''Describes three receptors'''

    def initialise(self):
        x = self.x
        if x.shape != (3, 2):
            raise ValueError
        
        D = np.log(norm(np.array([x[1]-x[2], x[0]-x[2], x[0]-x[1]]), axis=-1)/self.eps)
        self.fixed = -np.array([D[0]*(-D[0]+D[1]+D[2]), D[1]*(D[0]-D[1]+D[2]), D[2]*(D[0]+D[1]-D[2])])
        self.Gmat = np.array([[-2*D[0],        D[0]+D[1]-D[2],  D[0]-D[1]+D[2]],
                              [D[0]+D[1]-D[2], -2*D[1],         -D[0]+D[1]+D[2]],
                              [D[0]-D[1]+D[2], -D[0]+D[1]+D[2], -2*D[2]]])
        self.delta = D.sum()**2 - 4*(D[0]*D[1]+D[0]*D[2]+D[1]*D[2])
        
    def _evaluate(self, x0):
        return ((self.fixed + np.dot(self.G(x0[:, np.newaxis], self.x.T), self.Gmat)/2.)/self.delta).T
        
