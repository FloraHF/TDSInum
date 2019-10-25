import numpy as np
from math import pi, sin, cos, tan, acos
from scipy.optimize import fmin, minimize_scalar, minimize

from Config import Config
from coords import phy_to_thtalpha, phy_to_thtd

vd = Config.VD
vi = Config.VI
w = vi/vd   # velocity ratio
r = Config.CAP_RANGE                                        # capture range
LB = 2*acos(1/w)                                                # lower bound of phi_f
A = vd*tan(LB/2)/r

def evelope_circle():

    x0 = -r/tan(LB/2)
    y0 = -r
    rd = r/tan(LB/2)
    ri = r/sin(LB/2)

    xds, xis = [], []

    for t1 in np.linspace(0, 2*pi/A, 50):
        xd1 = x0 + rd*cos(A*t1+LB)
        yd1 = y0 + rd*sin(A*t1+LB)
        xi = x0 + ri*cos(A*t1+LB/2)
        yi = y0 + ri*sin(A*t1+LB/2)
        xds.append([xd1, yd1])
        xis.append([xi, yi])

    return np.asarray(xds), np.asarray(xis)


def evelope_barrier_state(t, T=1, gmm=LB/2):
    
    assert t >= 0
    
    if t >= T:
        t1 = T
        t2 = t - T
    else:
        t1 = t
        t2 = 0
    
    x0 = -r/tan(LB/2)
    y0 = -r
    rd = r/tan(LB/2)
    ri = r/sin(LB/2)

    xd1 = x0 + rd*cos(A*t1+LB) - vd*sin(A*t1+LB)*t2
    yd1 = y0 + rd*sin(A*t1+LB) + vd*cos(A*t1+LB)*t2
    xi = x0 + ri*cos(A*t1+LB/2) - vi*sin(A*t1+LB/2)*t2
    yi = y0 + ri*sin(A*t1+LB/2) + vi*cos(A*t1+LB/2)*t2
    xd2 = 0
    yd2 = r + vd*t
    
    d = yd2
    xd2 = d*sin(2*gmm - LB)
    yd2 = d*cos(2*gmm - LB)
    
    x = np.array([xd1, yd1, xd2, yd2, xi, yi])
    
    angles = phy_to_thtalpha(x)
    ds = phy_to_thtd(x)
    if (0<angles[0]<=pi/2) and (0<angles[1]<=pi/2) and (0<angles[2]< 2*pi):
        return x, angles, ds
    else:
        return x, None, None
    
def natural_barrier_state(t, delta, gmm):
    x = np.zeros(6,)
    x[0] = -(vd*t + r)*cos(gmm)
    x[1] = -(vd*t + r)*sin(gmm)
    x[2] = -(vd*t + r)*cos(gmm)
    x[3] =  (vd*t + r)*sin(gmm)
    x[4] = - vi*t*cos(delta)
    x[5] = - vi*t*sin(delta)

    angles = phy_to_thtalpha(x)
    ds = phy_to_thtd(x)
    if (0<angles[0]<=pi/2) and (0<angles[1]<=pi/2) and (0<angles[2]< 2*pi):
        return x, angles, ds
    else:
        return x, None, ds


def get_max_T(gmm):

    def mismatch(T):
        
        tht = A*T

        r1 = tht/sin(tht + LB)
        r2 = (tan(LB/2) + tan((2*pi - tht - LB)/2))/sin(2*gmm+tht)

        return (r1 - r2)**2
    
    assert gmm >= LB/2

    res = minimize_scalar(mismatch, bounds=[(2*pi-2*gmm)/A, (2*pi-LB)/A], method='bounded')
    Tmax = res.x
        
    return Tmax


def get_max_t(gmm, T):

    assert (2*pi - 2*gmm)/A <= T <= get_max_T(gmm)
    assert gmm >= LB/2

    def slope_mismatch(t):
        
        x, _, _ = evelope_barrier_state(T+t, T=T, gmm=gmm)
        
        D1 = np.array([x[0], x[1]])
        D2 = np.array([x[2], x[3]])
        I = np.array([x[4], x[5]])
        D1_I = (D1 - I)/np.linalg.norm(D1 - I)
        D2_I = (D2 - I)/np.linalg.norm(D2 - I)
        
        s_I_D1 = D1_I[1]/D1_I[0]
        s_I_D2 = D2_I[1]/D2_I[0]
        
    #    print('%.2f, %.2f, %.2f' %(s_I_D1, s_I_D2, (s_I_D1 - s_I_D2)**2))
        return (s_I_D1 - s_I_D2)**2

    dt = (2*pi-LB)/A - T
    res = minimize_scalar(slope_mismatch, bounds=[0, dt], method='bounded')
    tmax = res.x
        
    return tmax


def dwin_SS_traj_hh(delta, T, t=20, Nt=20, gmm=LB/2):

    assert gmm >= LB/2
    assert 0 <= delta <= gmm - LB/2

    Tmax = get_max_T(gmm)
    assert T <= Tmax

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

        if T < (2*pi-2*gmm)/A:
            ts = np.linspace(0, t, Nt)

        else:
            tmax = get_max_t(gmm, T)
            ts = np.linspace(0, T+tmax, int(Nt*(T+tmax)/t))
        for t in ts:
            x, s, d = evelope_barrier_state(t, T=T, gmm=gmm)
            if s is not None:
                xs.append(x)
                ss.append(s)
                ds.append(d)

    return np.asarray(xs), np.asarray(ss), np.asarray(ds), ts


# import matplotlib.pyplot as plt

# n = 4
# gmms = np.linspace(LB/2, pi-LB, n)
# Tus = np.zeros(n)
# Tls = np.zeros(n)

# for i, gmm in enumerate(gmms):
#     Tus[i] = get_max_T(gmm)
#     Tls[i] = 2*pi - 2*gmm
#     m = 20
#     TT = np.linspace(Tls[i], Tus[i], m)
#     ts = np.zeros(m)
#     fig, ax = plt.subplots()
#     for k, T in enumerate(TT):
#         ts[k] = get_max_t(gmm, T)
#     ax.plot(TT, ts)

# plt.show()

