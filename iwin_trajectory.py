import numpy as np
from math import pi, sin, cos, tan, acos, asin, atan2
import scipy.linalg

from Config import Config
from coords import phy_to_thtalpha, phy_to_thtd

vd = Config.VD
vi = Config.VI


def iwin_SS_straight_traj(tht, delta=0., L=300., nT=50, length=0.5):
    
    def get_init(tht, delta, L):
    
        P1 = np.array([-L, 0.])
        P2 = np.array([ L, 0.])

        r = L/sin(tht)
        Y = L/tan(tht)
        x = r*cos(delta)
        y = r*sin(delta) + Y
        P = np.array([x, y])

        l1 = np.linalg.norm(P-P1)
        l2 = np.linalg.norm(P-P2)
        cpsi = sin(tht)
        spsi = cos(tht) - l1/l2
        psi = atan2(spsi, cpsi)

        P1_P = P - P1
        P2_P = P - P2
        P_P2 = P2 - P
        phi_1 = atan2(P1_P[1], P1_P[0]) - pi/2
        phi_2 = atan2(P2_P[1], P2_P[0]) + pi/2
        psi = atan2(P_P2[1], P_P2[0]) + psi
        
        return P1, P2, P, phi_1, phi_2, psi
    
    p1, p2, p, phi_1, phi_2, psi = get_init(tht, delta, L)
    
    xs = [np.concatenate((p1, p2, p))]
    ss = [phy_to_thtalpha(xs[0])]
    ds = [phy_to_thtd(xs[0])]
    times = [0]
    
    dt_s = L*length/vd/nT

    dt_l = 0.98*L/vd
    
    for t in range(nT):
        if t == 0:
            dt = dt_l
        else:
            dt = dt_s

        p1 += vd*np.array([cos(phi_1), sin(phi_1)])*dt
        p2 += vd*np.array([cos(phi_2), sin(phi_2)])*dt
        p  += vi*np.array([cos(psi),   sin(psi)])*dt
        
        x = np.concatenate((p1, p2, p))
        
        angles = phy_to_thtalpha(x)
        if (0<angles[0]<pi/2) and (0<angles[1]<pi/2) and (0<angles[2]< 2*pi):
            xs.append(x)
            ss.append(angles)
            ds.append(phy_to_thtd(x))
            times.append(times[-1]+dt)
        else:
            break

#        print('[%.2f, %.2f], [%.2f, %.2f], [%.2f, %.2f], %.2f'%(p1[0], p1[1], p2[0], p2[1], p[0], p[1], ss[-1][-1]))
    outid = []
    return np.asarray(xs[3:-3]), np.asarray(ss[3:-3]), np.asarray(ds[3:-3]), np.asarray(times[3:-3])-times[3]