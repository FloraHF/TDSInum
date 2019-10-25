import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import matplotlib.tri as mtri
from matplotlib import rc

import numpy as np
from math import pi, tan, acos

from Config import Config
LB = 2*acos(Config.VD/Config.VI)

from hhtrajectory import dwin_SS_traj_hh, get_max_T
from iwin_trajectory import iwin_SS_straight_traj
from plotter import plot_traj_phy, plot_traj_thtalpha, plot_state_space_boundary, plot_sampled_SS, plot_iwin_bound
from plotter import plot_dominant, plot_zero_range_barrier, plot_zha_barrier, plot_flora_barrier
from sampler import sample_SS

def dwin_plots(delta, T, t=10, Nt=20, gmm=LB/2):
    fig, ax = plt.subplots()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    xs, ss, ds, times = dwin_SS_traj_hh(delta, T, t=t, Nt=Nt, gmm=gmm)
    plot_traj_phy(ax, xs[::-1], skip=10)

    plt.xlabel(r'$x(m)$', fontsize=16)
    plt.ylabel(r'$y(m)$', fontsize=16)
    plt.title(r'$\theta=\pi-2\gamma$, $\delta=0.3\pi$', fontsize=16)

    ax.axis('equal')
    plt.grid()
    plt.show()

def iwin_plots():
    fig, ax = plt.subplots()
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
#    xs, ss = iwin_SS_straight_traj(tht=pi-LB, delta=0.3*pi, length=0.4)
    xs, ss, ds, times = iwin_SS_straight_traj(tht=pi-LB+0.2, delta=0.3*pi, length=0.4)
    plot_traj_phy(ax, xs, skip=10)

    plt.xlabel(r'$x(m)$', fontsize=16)
    plt.ylabel(r'$y(m)$', fontsize=16)
    plt.title(r'$\theta=\pi-2\gamma$, $\delta=0.3\pi$', fontsize=16)

    ax.axis('equal')
    plt.grid()
    plt.show()

def SS_plots(gmms, theta):
    fig = plt.figure()
    ax = plt.gca(projection='3d')
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    colors = "brg"
    for i, gmm in enumerate(gmms):
        print(gmm)
        value, xs, ss, ds = sample_SS(gmm=gmm)
        # print(ss)
        if i == 0:
            barrier=True
        else:
            barrier=False
        plot_sampled_SS(ax, ss, color=colors[i], label=r'$\gamma=%.2f^\circ$'%gmm, barrier=barrier)
    # plot_state_space_boundary(ax)
    
#    xs, ss = sample_iwin_SS(tht=theta)
    # plot_sampled_SS(ax, ss, color=colors[-1], label=r'$\theta=%.2f^\circ$'%(2*pi - theta))

#    plot_dominant(ax)
    
    ax.set_xlabel(r'$\alpha_1$', fontsize=16)
    ax.set_ylabel(r'$\alpha_2$', fontsize=16)
    ax.set_zlabel(r'$\theta$', fontsize=16)
#    ax.legend()
    plt.title(r'SSs', fontsize=16)
    
    plt.show()
    
def plot_compare():

    fig, ax = plt.subplots()
    plt.rc('text', usetex=True)
    plot_zero_range_barrier(ax, gmm=LB/2-0.2, color='r')
    plot_zero_range_barrier(ax, gmm=LB/2+0.01, color='slateblue')
    plot_zero_range_barrier(ax, gmm=LB/2+0.1, color='navy')
    plot_zero_range_barrier(ax, gmm=LB/2+0.3, color='blue')
    plot_zero_range_barrier(ax, gmm=LB/2+0.5, color='royalblue')
    plot_zero_range_barrier(ax, gmm=LB/2+0.7, color='cornflowerblue')
    plot_zero_range_barrier(ax, gmm=LB/2+0.9, color='lightsteelblue')
    plot_zero_range_barrier(ax, gmm=pi/2, color='lightsteelblue')

    plot_zha_barrier(ax)
    plot_flora_barrier(ax)

    plt.axis('equal')
    plt.grid()
    ax.set_xlabel(r'$x$', fontsize=16)
    ax.set_ylabel(r'$y$', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
#    ax.legend()
    plt.title(r'Barrier', fontsize=16)
    
    plt.show()
    
# SS_plots([LB/2, LB/2+0.8], pi-LB+0.05)  

# delta_max = 0.1
# gmm = LB/2+delta_max
# Tmax = get_max_T(gmm) 
# dwin_plots(0.3*delta_max, 0.6*Tmax, gmm=gmm)


# iwin_plots()
plot_compare()