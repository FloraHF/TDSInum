import numpy as np
from copy import deepcopy
from math import asin, acos, sin, cos, tan, pi, atan2, copysign

from Config import Config
from Value import ValueFunc

vd = Config.VD
vi = Config.VI
l = Config.CAP_RANGE                                        # capture range
LB = Config.LB                                               # lower bound of phi_f
####################################################################################
################################### ENVIRONMENT ####################################

def sign(x):
    return copysign(1, x)

class InitialLocationGenerator(object):

    def __init__(self,  z_range=[1., 5.],
                        x_range=[-6., 6.],
                        y_range=[0., 6.]):

        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range

    def generate(self):
        x = np.random.uniform(self.x_range[0], self.x_range[1])*l
        y = np.random.uniform(self.y_range[0], self.y_range[1])*l
        z = np.random.uniform(self.z_range[0], self.z_range[1])*l

        return [np.array([-z, 0]), np.array([z, 0]), np.array([x, y])]

class TDSIGame():
    def __init__(self, game_name='2DSI', x_0=Config.X0s,
                 vfunc_dir='learned_v/'):
        
        self.game = game_name
        self.dcount = 2
        self.icount = 1
        self.pcount = self.dcount + self.icount
        self.l = Config.CAP_RANGE
        self.x_generator = InitialLocationGenerator()
        # self.Vfn = ValueFunc(read_dir=vfunc_dir)
        
        self.x_0 = x_0
        self._p = [None, None, None]
        self.time = 0
        self.dt = Config.TIME_STEP

        self.defenders = [Player(self, 'defender', self.x_0[0], self.dt), 
                          Player(self, 'defender', self.x_0[1], self.dt)]
        self.intruders = [Player(self, 'intruder', self.x_0[2], self.dt)]
        self.players = self.defenders + self.intruders
        self.update_vec()
        self.init_D12 = self.D1_D2
        
        self.done = False
        
    def is_captured(self):
        cap = False
        for d in self.defenders:
            dcap = np.linalg.norm(d.x - self.intruders[0].x) < self.l
            cap = cap or dcap
        return cap
    
    def is_passed(self):
        tht = self.get_tht()
        if tht > 2*pi - LB:
            return True

    def is_rotate(self):
        rot = atan2(np.cross(self.init_D12, self.D1_D2)[-1], np.dot(self.init_D12, self.D1_D2))
        if pi - abs(rot) < 1e-1:
            return True
    
    def update_vec(self):
        self.D2_I = np.concatenate((self.intruders[0].x - self.defenders[1].x, [0]))
        self.D1_I = np.concatenate((self.intruders[0].x - self.defenders[0].x, [0]))
        self.D1_D2 = np.concatenate((self.defenders[1].x - self.defenders[0].x, [0]))

    def get_x(self):
        return np.concatenate([p.x for p in self.players])

    def get_tht(self):

        k1 = atan2(np.cross(self.D1_D2, self.D1_I)[-1], np.dot(self.D1_D2, self.D1_I)) # angle between D1_D2 to D1_I
        k2 = atan2(np.cross(self.D1_D2, self.D2_I)[-1], np.dot(self.D1_D2, self.D2_I)) # angle between D1_D2 to D2_I
        tht = k2 - k1
        
#        if self.cross = True:
        if k1 < 0:
            tht += 2*pi

        return tht

    def get_thtd(self):

        d1 = max(np.linalg.norm(self.D1_I), self.l)      # state: alpha_1
        d2 = max(np.linalg.norm(self.D2_I), self.l)      # state: alpha_2

        return np.array([d1, d2, self.get_tht()])

    def get_thtalpha(self):

        alpha_1 = asin(np.clip(self.l/np.linalg.norm(self.D1_I), -1, 1))      # state: alpha_1
        alpha_2 = asin(np.clip(self.l/np.linalg.norm(self.D2_I), -1, 1))      # state: alpha_2

        return np.array([alpha_1, alpha_2, self.get_tht()])

    def opt_strategy(self, x):

        d = self.get_thtd()
        phi, psi = self.Vfn.evaluate_policy(d)
        # print(self.Vfn.evaluate_gradient(d), d)

        if l-d[0] < 0.2:
            psi = pi - d[2] - LB/2
            phi[0] = 0
        elif l-d[1] < 0.2:
            psi = -(pi - LB/2)
            phi[1] = 0

        return np.concatenate((phi, [psi]))

    def adjust_strategy(self, p):
        d = self.get_thtd()
        close = self.dt*Config.VD*1.3
        if abs(l-d[0]) < close and abs(l-d[1]) < close: # in both range
            p[0] = 0
            p[1] = 0
            p[2] = -d[2]/2
        elif abs(l-d[0]) < close: # in D1's range
            print('close')
            p[0] = 0.99*self._p[0]
            psi = acos(vd*cos(p[0])/vi)
            psi = -abs(psi)
            p[2] = pi - d[2] + psi
            # p[2] = pi - d[2] - (-p[0] + LB/2 - 0.6)
        # elif abs(r-x[1]) < 0.005*r:
        #     p[2] = -(pi - LB/2)
        #     p[1] = 0

        return p

    def step(self, act_n):

        if self._p[0] is not None:
            action_n = deepcopy(self.adjust_strategy(act_n))
        else:
            action_n = deepcopy(act_n)
        self._p = deepcopy(action_n)

        action_n[0] += atan2(self.D1_I[1], self.D1_I[0])
        action_n[1] += atan2(self.D2_I[1], self.D2_I[0])
        action_n[2] += atan2(-self.D2_I[1], -self.D2_I[0])
        
        for player, action in zip(self.players, action_n):
            player.step(action)
            self.update_vec()
        self.time += self.dt
        
        done = False
        info = 'free'
        if self.is_captured():
            done = True
            info = 'captured'
        elif self.is_passed():
            done = True
            info = 'passed'
        elif self.is_rotate():
            done = True
            info = 'rotated'
        self.done = done

        return self.get_x(), self.get_thtalpha(), self.get_thtd(),  done, info
    
    def advance(self, n_steps, policies=[None, None, None]):
        for i, policy in enumerate(policies):
            if policy is None:
                policies[i] = self.opt_strategy
        xs = [self.get_x()]
        ss = [self.get_thtalpha()]
        ds = [self.get_thtd()]
        times = [self.time]
        for t in range(n_steps):
            actions = []
            for i, policy in enumerate(policies):
                act = policy(xs[-1])
                actions.append(act[i])
            x, s, d, done, info = self.step(actions)
            xs.append(x)
            ss.append(s)
            ds.append(d)
            times.append(self.time)
            # print(x[4:])
            if done:
                print(info)
                break
        return np.asarray(xs), np.asarray(ss), np.asarray(ds), np.asarray(times), info

    def reset(self, xs=None, rand=False):
        
        if xs is None:
            if rand:
                xs = self.x_generator.generate()
            else:
                xs = Config.X0s
                
        for p, x in zip(self.players, xs):
            p.reset(x)
        self.update_vec()
        self.time = 0
        self.done = False
        info = 'reseted'

        return self.get_x(), self.get_thtalpha(), self.get_thtd(), self.done, info

####################################################################################
#################################### PLAYERS #######################################

class Player(object):

    def __init__(self, env, role, x, dt):

        self.env = env
        self.role = role
        self.dt = dt
        self.x = x
        
        if role == 'defender':
            self.v = Config.VD
        elif role == 'intruder':
            self.v = Config.VI
    
    def step(self, action):
        self.x += self.dt * self.v * np.array([cos(action), sin(action)])
    
    def reset(self, x):
        self.x = deepcopy(x)
