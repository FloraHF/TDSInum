import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
from math import pi, cos, sin, tan, sqrt, acos
import csv

from Config import Config

vd = Config.VD
vi = Config.VI
w = vi/vd   # velocity ratio
LB = 2*acos(1/w)                                                # lower bound of phi_f
a = w

l = Config.CAP_RANGE                                        # capture range

def plot_state_space_boundary(ax):
    """ plot the boundary of the state space
    """
    
    yy, zz = np.meshgrid(np.linspace(0, pi/2, 2), np.linspace(0, 2*pi, 2))
    xx = np.zeros(np.shape(yy))
    ax.plot_wireframe(xx, yy, zz, color='royalblue')
    ax.plot_surface(xx, yy, zz, color='cornflowerblue', alpha=0.2)
    
    xx = pi/2*np.ones(np.shape(yy))
    ax.plot_wireframe(xx, yy, zz, color='lightsteelblue')
    ax.plot_surface(xx, yy, zz, color='cornflowerblue', alpha=0.1)
    
    xx, zz = np.meshgrid(np.linspace(0, pi/2, 2), np.linspace(0, 2*pi, 2))
    yy = np.zeros(np.shape(yy))
    ax.plot_wireframe(xx, yy, zz, color='lightsteelblue')
    ax.plot_surface(xx, yy, zz, color='cornflowerblue', alpha=0.2)
    
    yy = pi/2*np.ones(np.shape(xx))
    ax.plot_wireframe(xx, yy, zz, color='lightsteelblue')
    ax.plot_surface(xx, yy, zz, color='cornflowerblue', alpha=0.1)
    
    xx, yy = np.meshgrid(np.linspace(0, pi/2, 2), np.linspace(0, pi/2, 2))
    zz = np.zeros(np.shape(xx))
    ax.plot_wireframe(xx, yy, zz, color='lightsteelblue')
    ax.plot_surface(xx, yy, zz, color='cornflowerblue', alpha=0.2)
    
    zz = 2*pi*np.ones(np.shape(xx))
    ax.plot_wireframe(xx, yy, zz, color='lightsteelblue')
    ax.plot_surface(xx, yy, zz, color='cornflowerblue', alpha=0.1)
    
    ax.set_xlim3d(-0.1, pi/2)
    ax.set_ylim3d(-0.1, pi/2)
    ax.set_zlim3d(-0.1, 2*pi)
    
    ax.set_xlabel('alpha_1')
    ax.set_ylabel('alpha_2')
    ax.set_zlabel('theta')
                  
def plot_dominant(ax):
    """ plot B2
    """
    xxx, yyy = np.meshgrid(np.linspace(0, pi/2, 3), np.linspace(0, pi/2, 3))
    zzz = np.zeros(np.shape(yyy))
    for i, (xx, yy) in enumerate(zip(xxx, yyy)):
        for j, (x, y) in enumerate(zip(xx, yy)):
            zzz[i, j] = pi - LB + (x + y)
    
    ax.plot_wireframe(xxx, yyy, zzz, color='turquoise')
    ax.plot_surface(xxx, yyy, zzz, color='turquoise', alpha=0.6)
        
        
def plot_target(ax, n=50):

    def target(t):
        a, b = Config.TARGET_RADIUS, Config.TARGET_RADIUS
        n = np.array([b*cos(t), a*sin(t)])
        n = n/sum(n*n)**0.5
        return np.array([a*cos(t), b*sin(t)]), n

    def sample_target(n=50):

        ts = np.linspace(0, 2*pi, num=n)
        x = np.array([[0.0, 0.0] for _ in range(n)])
        for i, t in enumerate(ts):
            x[i], _ = target(t)
        return x

    tg = sample_target(n)
    ax.plot(tg[:,0], tg[:,1], 'k', label='target')

def plot_capture_ring(ax, loc_D, n=50, line_style=(0, ()), color='b', label=None):

    def capture_ring(t, loc):
        return np.array([loc[0]+l*cos(t), loc[1]+l*sin(t)])

    def sample_capture_ring(loc_D, n=50):

        ts = np.linspace(0, 2*pi, num=n)
        x = np.array([[0.0, 0.0] for _ in range(n)])
        for i, t in enumerate(ts):
            x[i] = capture_ring(t, loc_D)
        return x

    cp = sample_capture_ring(loc_D)
    ax.plot(cp[:,0], cp[:,1], color=color, linestyle=line_style)

def plot_traj_phy(ax, traj, line_style=(0, ()), dlabel='', ilabel='', skip=1, reverse=False, connect=False, marker=False):
    
    if reverse:
        traj = traj[::-1]
    
    x_p1s = [state[0] for state in traj]
    y_p1s = [state[1] for state in traj]
    x_p2s = [state[2] for state in traj]
    y_p2s = [state[3] for state in traj]
    x_es  = [state[4] for state in traj]
    y_es  = [state[5] for state in traj]

    ax.plot(x_p1s[-1], y_p1s[-1], 'b*')
    ax.plot(x_p2s[-1], y_p2s[-1], 'b*')
    ax.plot(x_es[-1],  y_es[-1],  'r*')

    if connect:
        for i, (x_p1, y_p1, x_p2, y_p2, x_e, y_e) in enumerate(zip(x_p1s, y_p1s, x_p2s, y_p2s, x_es, y_es)):
            if (i%skip == 0 and i < len(x_p1s)-skip) or i == len(x_p1s)-1:
                ax.plot([x_p1, x_e], [y_p1, y_e], 'b--', label=None, zorder=1)
                ax.plot([x_p2, x_e], [y_p2, y_e], 'b--', label=None, zorder=1)

    if marker:
        ax.plot(x_p1s, y_p1s, '.', markevery=skip, color='b', linestyle=line_style, label='Defender, '+dlabel)
        ax.plot(x_p2s, y_p2s, '.', markevery=skip, color='b', linestyle=line_style, label=None)
        ax.plot(x_es,  y_es,  '.', markevery=skip, color='r', linestyle=line_style, label='Intruder, '+ilabel, zorder=100)
    else:
        ax.plot(x_p1s, y_p1s, color='b', linestyle=line_style, label='Defender, '+dlabel)
        ax.plot(x_p2s, y_p2s, color='b', linestyle=line_style, label=None)
        ax.plot(x_es,  y_es,  color='r', linestyle=line_style, label='Intruder, '+ilabel, zorder=101)
#
    plot_capture_ring(ax, np.array([x_p1s[-1], y_p1s[-1]]), color='b', line_style=line_style, label=None)
    plot_capture_ring(ax, np.array([x_p2s[-1], y_p2s[-1]]), color='b', line_style=line_style, label=None)

#    plot_target(ax)

def plot_traj_thtalpha(ax, traj, color='k', skip=10, label=None, fake=False, zorder=None):
    
    a1  = [x[0] for x in traj]
    a2  = [x[1] for x in traj]
    tht = [x[2] for x in traj]
    
    if zorder is not None:
        ax.plot(a1, a2, tht, '.', linestyle=(0, ()), linewidth=5.0, color=color, markevery=skip, label=label, zorder=zorder)
    else:
        ax.plot(a1, a2, tht, '.', linestyle=(0, ()), color=color, markevery=skip, label=label)
            
def plot_sampled_SS(ax, trajs, color='b', label=None, barrier=False):
    x, y, z = trajs[0][:,0], trajs[0][:,1], trajs[0][:,2]
    for traj in trajs[1:]:
        x = np.concatenate((x, traj[:,0]))
        y = np.concatenate((y, traj[:,1]))
        z = np.concatenate((z, traj[:,2]))
    nb = np.array([[0, 0, pi-LB], [pi/2, pi/2, 2*pi-LB]])
    ax.plot_trisurf(x, y, z, color=color, linewidth=0.2, alpha=0.6, label=None)
    # ax.scatter([-100], [-100], [0], c=color, edgecolors='none', label=label)
    # plot_traj_thtalpha(ax, nb, color='g', skip=5, zorder=2, label='The natural barrier')
    for i, traj in enumerate(trajs):
        if i == (len(trajs)-1) and barrier:
            pass
            # plot_traj_thtalpha(ax, traj, color='g', skip=5, zorder=1, label='The envelope line')
        else:
            if i % 20 == 0:
                plot_traj_thtalpha(ax, traj, color=color, skip=10)
    ax.set_zlim(0, 2*pi)
    ax.set_xlim(0, pi/2)
    ax.set_ylim(0, pi/2)
#    plot_traj_thtalpha(ax, trajs[-1], color=color)

def plot_iwin_bound(ax, tht=pi-LB, L=100, N=50):
    
    P1 = np.array([-L, 0.])
    P2 = np.array([ L, 0.])
    r = L/sin(tht)
    Y = L/tan(tht)
    
    gmm = pi/2.9
    a1_l = gmm
    a2_l = pi - tht - gmm
    d_l = 2*L/sin(tht)*sin(a2_l)
    x_l = d_l*cos(a1_l) - L
    y_l = d_l*sin(a1_l)
    
    a1_u = gmm - tht
    a2_u = pi - gmm
    d_u = 2*L/sin(tht)*sin(a2_u)
    x_u = d_u*cos(a1_u) - L
    y_u = d_u*sin(a1_u)
    print(a1_u)
    print(y_u)
    
    deltas = np.linspace(-(pi/2-tht), 1.5*pi-tht, N)
    xu, yu, xd, yd = [], [], [], []

    for delta in deltas:
        x = r*cos(delta)
        y = r*sin(delta) + Y
        xu.append(x)
        yu.append(y)
        xd.append(x)
        yd.append(-y)
    
    ax.plot(xu, yu)
    ax.plot(xd, yd)
    ax.plot(x_l, y_l, marker='.', color='g')
#    ax.plot(x_u, y_u, marker='.', color='r')
 
def plot_zha_barrier(ax, L=10):
    
    def contourfn(x, y):

        D1 = np.array([-L, 0, 0])
        D2 = np.array([L, 0, 0])
        P = np.array([x, y, 0])
        
        d1 = np.linalg.norm(P-D1)
        d2 = np.linalg.norm(P-D2)
        d = np.linalg.norm(D1-D2)
        
        return d - (d1 + d2)/a
        
    X, Y = np.meshgrid(np.linspace(-1.5*L, 1.5*L), np.linspace(0, 1.5*L))
    Z = np.zeros(X.shape)
    for i, (xx, yy) in enumerate(zip(X, Y)):
        for j, (x, y) in enumerate(zip(xx, yy)):
            Z[i, j] = contourfn(x, y)
    ax.contour(X, Y, Z, [0], zorder=10)

def plot_flora_barrier(ax, L=10):
    
    def contourfn(x, y):
#        aa = 1/a
#        return y**2*aa**4 + (x**2+y**2)*a**2*(1-a**2)-L**2*(1-a**2)
        return y**2 - (a**2-1)*(a**2*L**2 - x**2 - y**2)
    X, Y = np.meshgrid(np.linspace(-1.5*L, 1.5*L), np.linspace(0, 1.5*L))
    Z = np.zeros(X.shape)
    for i, (xx, yy) in enumerate(zip(X, Y)):
        for j, (x, y) in enumerate(zip(xx, yy)):
            Z[i, j] = contourfn(x, y)
    ax.contour(X, Y, Z, [0], zorder=10)    
    
def plot_zero_range_barrier(ax, L=10, gmm=LB/2, color='cornflowerblue', line_style=(0, (5, 3))):
    x0 = 0
    y0 = -L/tan(gmm)
    r = a*L/sin(gmm)
    thts = np.linspace(pi/2-gmm, pi/2+gmm)
    xs, ys = [], []
    for tht in thts:
        x = r*cos(tht) + x0
        y = r*sin(tht) + y0
        xs.append(x)
        ys.append(y)
    ax.plot(xs, ys, color=color, linestyle=line_style)
    ax.plot([L], [0], marker='.', color='b')
    ax.plot([-L], [0], marker='.', color='b')
    
def plot_fitted_SS(ax, fn, gmm, color='k', xlim=10):
    
#    x = np.linspace(0, pi/2, 25)
#    y = np.linspace(0, pi/2, 25)
#    X, Y = np.meshgrid(x, y)
#    Z = np.zeros(X.shape)
#    
##    fig_, ax_ = plt.subplots()
#    
#    zs = np.linspace(0, 2*pi-LB, 10)
#    for k, z in enumerate(zs):
#        for i, (xx, yy) in enumerate(zip(X, Y)):
#            for j, (xxx, yyy) in enumerate(zip(xx, yy)):
#                Z[i,j] = fn(np.array([xxx, yyy, z])) - gmm
#        cs = ax.contour(X, Y, Z+z, [z], zdir='z')
        
#        path = cs.allsegs[0][0]
#        
#        if k == 0:
#            a1 = path[:,0]
#            a2 = path[:,1]
#            tht = z*np.ones(np.shape(path)[0], )
#        else:
#            a1 = np.concatenate((a1, path[:,0]))
#            a2 = np.concatenate((a2, path[:,1]))
#            tht = np.concatenate((tht, z*np.ones(np.shape(path)[0], )))
#    
#    ax.plot_trisurf(a1, a2, tht, linewidth=0.2, antialiased=True)
    x = np.linspace(0, xlim, 25)
    z = np.linspace(0, 2*pi, 25)
    X, Z = np.meshgrid(x, z)
    Y = np.zeros(X.shape)

    ys = np.linspace(0, xlim, 10)
    for y in ys:
        for i, (xx, zz) in enumerate(zip(X, Z)):
            for j, (xxx, zzz) in enumerate(zip(xx, zz)):
                Y[i,j] = fn(np.array([xxx, y, zzz])) - (2*pi - 2*gmm)
        cs = ax.contour(X, Y+y, Z, [y], zdir='y')

    y = np.linspace(0, xlim, 25)
    z = np.linspace(0, 2*pi, 25)
    Y, Z = np.meshgrid(y, z)
    X = np.zeros(X.shape)

    xs = np.linspace(0, xlim, 10)
    for x in xs:
        for i, (yy, zz) in enumerate(zip(Y, Z)):
            for j, (yyy, zzz) in enumerate(zip(yy, zz)):
                X[i,j] = fn(np.array([x, yyy, zzz])) - (2*pi - 2*gmm)
        cs = ax.contour(X+x, Y, Z, [x], zdir='x')
        
#    plt.close(fig_)

def plot_V_error(ax):
    train_loss, eval_loss = [], []
    with open('loss.csv', 'r') as f:
        readings = csv.reader(f)
        for rd in readings:
            train_loss.append(float(rd[0]))
            eval_loss.append(float(rd[1]))

    ax.plot(range(len(train_loss)), train_loss)
    ax.plot(range(len(eval_loss)), eval_loss)
    ax.grid()
