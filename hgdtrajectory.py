import numpy as np
from math import pi, sin, cos, tan, acos, asin, atan2, sqrt
from mpmath import ellipe, ellipf
from scipy.optimize import minimize_scalar

from Config import Config
from coords import phy_to_thtalpha, phy_to_thtd
from hhtrajectory import natural_barrier_state

vd = Config.VD
vi = Config.VI
w = vi/vd   # velocity ratio
l = Config.CAP_RANGE                                        # capture range
LB = 2*acos(1/w)                                                # lower bound of phi_f

def barrier_state(phi, tau, gmm):
    """ trajectory under opitmal play, initial locations should be on SS,
        reference frame: physical 6D frame, y is along ID_2 upon capture
        ----------------------------------------------------------------------------
        Input:
            phi: phi_1 in [lb, ub]                                      float
            tau: time spent on the straight line trajectory             float
            lb:  lower bound of phi.                                    float
                 lb=LB: at barrier,
                 lb>LB: other SS
        Output:
            x: 6D trajectory, order: D1, D2, I,                         np.array
        ----------------------------------------------------------------------------
    """

    assert 0 <= gmm <= pi
    
    def consts(phi):
        s, c = sin(phi/2), cos(phi/2)
        A, B = l/(w**2 - 1), sqrt(1 - w**2*c**2)
        return s, c, A, B

    def xtau(phi):
        out = np.zeros((6,))
        s, c, A, B = consts(phi)
        E = ellipe(asin(w*c), 1/w);
        F = ellipf(asin(w*c), 1/w);
        out[0] = - 2*A*(s**2*B + w*s**3)                                        # x_D1
        out[1] = A*(w*E + 3*w*c + 2*s*c*B - 2*w*c**3)                           # y_D1
        out[3] = 2*l*F/w - 3*w*E*A - 3*w*c*A                                    # y_D2
        out[4] = A*(B*(2*w**2*c**2 - w**2 - 1) - 2*w**3*s**3 + 2*w*(w**2-1)*s)  # x_I
        out[5] = A*(w*E + 2*w**2*s*c*B - 2*w**3*c**3 + w*c*(2+w**2))            # y_I
        return out

    s, c, A, B = consts(phi)

    xtemp = np.zeros((6,))
    xtemp[0] = -l*sin(LB) - 2*s*c*tau                                   # x_D1
    xtemp[1] =  l*cos(LB) + (2*c**2 - 1)*tau                            # y_D1
    xtemp[3] =  l         + tau                                         # y_D2
    xtemp[4] =            - (w**2*s*c + w*c*B)*tau                      # x_I
    xtemp[5] =            + (w**2*c**2 - w*s*B)*tau                     # y_I

    dx = xtau(phi) - xtau(LB)
    if dx[3] < 0:
        dx[3] = 0
    dt = dx[3]
    x = dx + xtemp
    t = dt + tau

    d = sqrt(x[2]**2 + x[3]**2)
    x[2] = d*sin(2*gmm - LB)
    x[3] = d*cos(2*gmm - LB)
    
    x = x*vd

    s = phy_to_thtalpha(x)
    d = phy_to_thtd(x)
    
    return x, s, d, t

def get_max_phi(gmm):
    """ obtain the maximum phi
        ----------------------------------------------------------------------------
        Input:
            lb:  lower bound of phi.                                            float
                 lb=LB: at barrier,
                 lb>LB: other SS
        Output:
            ub: (=res.x)                                                        float
                maximum phi is obtained when I, D2, D1 forms a straight line
        ----------------------------------------------------------------------------
    """

    def slope_mismatch(phi):
        x, _, _, _ = barrier_state(phi, 0, gmm)
        s_I_D1 = (x[1] - x[5])/(x[0] - x[4])
        s_I_D2 = (x[3] - x[5])/(x[2] - x[4])
        return (s_I_D1 - s_I_D2)**2

    res = minimize_scalar(slope_mismatch, bounds=[LB, 2*pi-LB], method='bounded')

    return res.x

def get_max_gmm():
    res = minimize_scalar(get_max_phi, bounds=[pi/2, pi], method='bounded')
    return res.x

# gmm_max = get_max_gmm()

def dist_DI(gmm, phi):
    
    x, _ = barrier_state(phi, 0, gmm)
    d = sqrt((x[2] - x[4])**2 + (x[3] - x[5])**2)
    
    return d
#
#fig, ax = plt.subplots()
#gmms = np.linspace(LB/2, LB-0.5*pi, 5)
#
#for gmm in gmms:
#    ds = []
#    phis = np.linspace(LB, get_max_phi(gmm))
#    for phi in phis:
#        ds.append(dist_DI(gmm, phi))
#    ax.plot(phis, ds)
#
#plt.grid()
#plt.show()

def d_dist_DI(gmm):
    
    phi_1 = LB
    phi_2 = pi + (LB - 2*gmm)
    
    x, _, _, _ = barrier_state(phi_1, 0, gmm)
    
    s, c = sin(phi_1/2), cos(phi_1/2)
    tht = phi_1 + acos(w*c)
    psi = tht - phi_1/2
    
    dx_i = w*cos(psi)
    dy_i = w*sin(psi)
    dx_2 = cos(phi_2)
    dy_2 = sin(phi_2)
    
    dd = (x[2] - x[4])*(dx_2 - dx_i) + (x[3] - x[5])*(dy_2 - dy_i)
    
    return dd

#gmms = np.linspace(0, pi, 50)
#d = []
#for gmm in gmms:
#    d.append(d_dist_DI(gmm))
#fig, ax = plt.subplots()
#ax.plot(gmms, d)
#ax.plot([LB/2, LB/2], [-5, 5])
#plt.grid()
#plt.show()

def get_min_gamma():

    def dd_square(gmm):
        return (d_dist_DI(gmm))**2
    
    res = minimize_scalar(dd_square, bounds=[0, LB/2], method='bounded')

    return res.x

# gmm_min = get_min_gamma()

# def dwin_SS_state(phi, t, delta=0, gmm=LB/2):
#     """ obtain the maximum phi
#         ----------------------------------------------------------------------------
#         Input:
#             gmmi: in range [LB, gmm]
#         Output:
#             x: state
#         ----------------------------------------------------------------------------
#     """
#     assert gmm >= LB/2
#     assert 0 <= delta <= gmm - LB/2
#     assert t >= 0
    
#     if gmm < LB/2 or (gmm >= LB/2 and delta == gmm - LB/2):
#         x, time = barrier_state(phi, t, gmm)
#     else:
#         x = np.zeros(6,)
#         x[0] = -(vd*t + l)*cos(gmm)
#         x[1] = -(vd*t + l)*sin(gmm)
#         x[2] = -(vd*t + l)*cos(gmm)
#         x[3] =  (vd*t + l)*sin(gmm)
#         x[4] = - vi*t*cos(delta)
#         x[5] = - vi*t*sin(delta)
#         time = t

#     angles = phy_to_thtalpha(x)
#     ds = phy_to_thtd(x)
#     if (0<angles[0]<=pi/2) and (0<angles[1]<=pi/2) and (0<angles[2]<= 2*pi):
#         return x, angles, ds, time
#     else:
#         return x, None, ds, time
    
def dwin_SS_traj(delta, phi, t=5, Nt=30, gmm=LB/2):
    
#    print('new trajectory')
    
    assert gmm >= LB/2
    assert 0 <= delta <= gmm - LB/2

    xs, ss, ds = [], [], []

    if (gmm - LB/2) - delta > 1e-6:
        ts = np.linspace(0, t, Nt)
        for t in ts:
            x, s, d = natural_barrier_state(t, delta, gmm)
            if s is not None:
                xs.append(x)
                ss.append(s)
                ds.append(d)
    else:
        phi = min(phi, get_max_phi(gmm))
        _, _, _, tmax = barrier_state(phi, 0, gmm)

        Np = int(Nt*(tmax/t))
        phis = np.linspace(LB, phi, Np*2+1)

        tau = max(t - tmax, 0)
        ts = []
        for p in phis:
            x, s, d, t = barrier_state(p, 0, gmm)
            # print(s)
            if s is not None:
                xs.append(x)
                ss.append(s)
                ds.append(d)
                ts.append(t)
        if tau > 0:
            taus = np.linspace(0, tau, Nt-Np)
            for tau in taus[1:]:
                x, s, d, t = barrier_state(phis[-1], tau, gmm)
                if s is not None:
                    xs.append(x)
                    ss.append(s)
                    ds.append(d)
                    ts.append(t)

    return np.asarray(xs), np.asarray(ss), np.asarray(ds), np.asarray(ts)


# x, s, d, t = dwin_SS_traj(0, LB/2, gmm=LB/2)
# print(barrier_state(LB, 0, LB))
# print(sqrt(1 - w**2*cos(LB/2)**2))