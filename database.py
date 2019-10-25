import numpy as np
from math import pi, acos

from Config import Config
r = Config.CAP_RANGE

class BoundaryStateGenerator(object):
    
    def __init__(self, frame='thtd'):
           
        self.gmm_lb = acos(Config.VD/Config.VI)
        self.gmm_ub = pi
        self.frame = frame
        
    def random_batch(self, batch_size):
        
        gmm = np.random.uniform(self.gmm_lb, self.gmm_ub, (batch_size, ))
        d1 = r*np.ones((batch_size, ))
        d2 = r*np.ones((batch_size, ))
        tht = 2*pi - 2*gmm
        if self.frame == 'thtd':
            batch = {'states': np.stack((d1, d2, tht), axis=1),
                     'values': tht.reshape((-1,1))}
        elif self.frame == 'thtalpha':        
            batch = {'states': np.stack((a1, a2, gmm), axis=1),
                     'values': gmm.reshape((-1,1))}
        
        return batch
    
class InnerStateGenerator(object):
    
    def __init__(self, frame='thtd'):

        r = Config.CAP_RANGE
        
        self.frame = frame
        self.states = []
        self.size = 0
        if self.frame == 'thtd':
            self.bds = [[r, 100*r], [r, 100*r], [0, 2*pi]]
        elif self.frame == 'thtalpha':
            self.bds = [[0, pi/2], [0, pi/2], [0, 2*pi]]
        self.ext = 0
        
    def random_batch(self, batch_size):
        
        tht = np.random.uniform(self.bds[2][0] - self.ext, self.bds[2][1] + self.ext, (batch_size,1))
        if self.frame == 'thtd':
            s1  = np.random.uniform(self.bds[0][0] - self.ext, self.bds[0][1] + self.ext, (batch_size,1))
            s2  = np.random.uniform(self.bds[1][0] - self.ext, self.bds[1][1] + self.ext, (batch_size,1))
        elif self.frame == 'thtalpha':
            s1  = np.random.uniform(self.bds[0][0] - self.ext, self.bds[0][1] + self.ext, (batch_size,1))
            s2  = np.random.uniform(self.bds[1][0] - self.ext, self.bds[1][1] + self.ext, (batch_size,1))
        
        batch = {'states': np.concatenate((s1, s2, tht), axis=1)}
    
        return batch

class StateValueBuffer(object):
    
    def __init__(self, max_size=Config.MAX_BUFFER_SIZE):
        self.values = []
        self.states = []
        self.max_size = max_size
        self.size = 0
        
    def add_state(self, value, state):
        if self.size > self.max_size:
            del self.values[0]
            del self.states[0]
            self.size -= 1
        self.values.append([value])
        self.states.append(state)
        self.size += 1
        # print(value, state)

    def add_state_pair(self, value, state):
        self.add_state(value, state)
        self.add_state(value, state[[1, 0, 2]])

    def add_traj(self, value, traj):
        for state in traj:
            self.add_state_pair(value, state)

    def add_surface(self, value, surface):
        for traj in surface:
            self.add_traj(value, traj)
        
    def random_batch(self, batch_size):
        if batch_size > self.max_size:
            batch_size = int(0.9*self.max_size)
        
        indices = np.random.randint(0, self.size, batch_size)
        v = np.asarray([self.values[k] for k in indices])
        s = np.asarray([self.states[k] for k in indices])
    
        return {'values': v, 'states': s}


                    
        