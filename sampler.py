import numpy as np
from math import pi, sin, cos, tan, ceil

from Config import Config
from hhtrajectory import get_max_T, dwin_SS_traj_hh
from hgdtrajectory import get_max_phi, dwin_SS_traj
from iwin_trajectory import iwin_SS_straight_traj

LB = Config.LB

def value(gmm):
    return 2*pi - 2*gmm

def sample_SS_traj(Nt=20):

    # sample gmm
    gmm = np.random.uniform(LB/2, 0.99*pi)

    # sample delta
    if np.random.uniform() <= 0.05:
        delta = gmm - LB/2
    else:
        delta = np.random.uniform(0, gmm-LB/2)

    # sample T
    T = np.random.uniform(0, get_max_T(gmm))

    # sample t
    if T < 2*pi - 2*gmm:
        t = 16
    else:
        t = np.random.uniform(0, get_max_t(gmm, T))

    x, s, d, _ = dwin_SS_traj_hh(delta, T, t=t, Nt=Nt, gmm=gmm)

    return value(gmm), d
    
def sample_terminal_states():
    
    r = Config.CAP_RANGE
    gmm = np.random.uniform(LB/2, 0.99*pi)
    tht = 2*pi - 2*gmm
    x = np.array([r, r, tht])
    
    return value(gmm), x

def sample_iwin_SS(tht=pi-LB, Nd=20, nT=200):

    assert pi-LB <= tht < pi
    deltas = np.linspace(pi/3, pi - pi/3, Nd)
    
    xs, ss, ds = [], [], []
    for delta in deltas:
        x, s, d, _ = iwin_SS_straight_traj(tht=tht, delta=delta, nT=nT)
        xs.append(x)
        ss.append(s)
        ds.append(d)
        
    return xs, ss, ds

def sample_SS(gmm=None, NT=10, Nd=10, tmax=20, Nt=20):
    if gmm is None:
        gmm = np.random.uniform(LB/2, LB/2+pi/2)
    
    xs, ss, ds = [], [], []
    
    # Tmax = get_max_T(gmm)
    phi_max = 0.95*get_max_phi(gmm)
    phis = np.linspace(LB, 0.91*phi_max, 20*NT)
    for phi in phis:
        # x, s, d, _ = dwin_SS_traj_hh(gmm-LB/2, T, t=tmax, Nt=Nt, gmm=gmm)
        x, s, d, _ = dwin_SS_traj(0, phi, t=20, Nt=30, gmm=gmm)
        # print(x)
        xs.append(x)
        ss.append(s)
        ds.append(d)
    if gmm > LB/2:
        deltas = np.linspace(0, gmm-LB/2, ceil(Nd*(gmm-LB/2)/(pi/2)))
        for delta in deltas:
            # x, s, d, _ = dwin_SS_traj_hh(delta, 0, t=tmax, Nt=Nt, gmm=gmm)
            x, s, d, _ = dwin_SS_traj(delta, phi, t=20, Nt=30, gmm=gmm)
            xs.append(x)
            ss.append(s)
            ds.append(d)
    
    return value(gmm), xs, ss, ds