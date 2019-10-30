import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

import numpy as np
from math import pi
import csv
import os

from Config import Config
from hhtrajectory import evelope_circle
from coords import phy_to_hds, traj_to_hds, circ_to_hds
from TDSIgame import TDSIGame
from plotter import plot_traj_phy, plot_fitted_SS, plot_state_space_boundary, plot_traj_thtalpha, plot_sampled_SS, plot_dominant
from hhtrajectory import dwin_SS_traj_hh, get_max_T, LB
from hgdtrajectory import dwin_SS_traj
from iwin_trajectory import iwin_SS_straight_traj
from sampler import sample_SS, sample_iwin_SS
from strategies import z_strategy, h_strategy, t_strategy, s_strategy, i_strategy, m_strategy


class Replay(object):

    def __init__(self, vfunc_dir='fitted_v/', res_dir='replay/'):

        self.vfunc_dir = vfunc_dir
        self.game = TDSIGame(vfunc_dir=vfunc_dir)
        self.res_dir = res_dir
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        self.prefix = vfunc_dir.split('_')[0] + '_'

    def conv(self):
        with open(self.vfunc_dir+'loss.csv', 'r') as f:
            loss = []
            reader = csv.reader(f)
            for row in reader:
                loss.append(float(row[0]))
        fig, ax = plt.subplots()
        plt.rc('text', usetex=True)
        
        ax.plot(np.arange(len(loss)), loss)
        plt.ylim([0,1])
        plt.grid()
        
        plt.xlabel(r'iteration')
        plt.ylabel(r'residual')
        plt.title(r'MSE', fontsize=12)

        plt.savefig(self.res_dir + '/' +'conv'+'.png')
        plt.close(fig)

    def Vcontour(self, v, xlim=10):
        fig = plt.figure()
        plt.rc('text', usetex=True)
        ax = plt.gca(projection='3d')
        plot_fitted_SS(ax, self.game.Vfn.evaluate_value, v, xlim=xlim)
        
        ax.set_xlabel(r'$d_1$', fontsize=16)
        ax.set_ylabel(r'$d_2$', fontsize=16)
        ax.set_zlabel(r'$\theta$')

        ax.set_xlim3d(0, xlim)
        ax.set_ylim3d(0, xlim)
        ax.set_zlim3d(0, 2*pi)

        plt.title(r'$V(x)=%.2f$'%v, fontsize=16)
        plt.savefig(self.res_dir + '/'+ self.prefix + '_V=%.2f'%v+'.png')
        plt.show()
        plt.close(fig)

    def _plot_alpha(self, ss):
        fig, ax = plt.subplots()
        ax.plot(ss[:,0], ss[:,1], '.-')
        ax.grid()
        plt.show()

    def _log_traj(self, xs, file_name):
        O, C = phy_to_hds(xs)
        xs = traj_to_hds(xs, O, C)
        with open(file_name, 'a') as f:
            for x in xs:
                f.write(','.join(list(map(str, x))) + '\n')

    def _plot_traj(self, xs_anl, xs_ply, fig_title, file_name, fontsize=16, dlabel='', ilabel=''):

        O, C = phy_to_hds(xs_anl)
        xs_anl = traj_to_hds(xs_anl, O, C)
        O, C = phy_to_hds(xs_ply)
        xs_ply = traj_to_hds(xs_ply, O, C)

        xmin = min(np.amin(xs_anl[:,[0, 2, 4]]), np.amin(xs_ply[:,[0, 2, 4]]))
        xmax = max(np.amax(xs_anl[:,[0, 2, 4]]), np.amax(xs_ply[:,[0, 2, 4]]))
        dx = (xmax - xmin)*0.2
        ymin = min(np.amin(xs_anl[:,[1, 3, 5]]), np.amin(xs_ply[:,[1, 3, 5]]))
        ymax = max(np.amax(xs_anl[:,[1, 3, 5]]), np.amax(xs_ply[:,[1, 3, 5]]))
        dy = (ymax - ymin)*0.3

        # trajectories in physical space
        fig = plt.figure()
        ax = fig.add_subplot(111, xlim=(xmin-dx, xmax+dx), ylim=(ymin-dy, ymax+dy))
        ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
        # plt.rc('text', usetex=True)
        
        plot_traj_phy(ax, xs_anl, line_style=(0, (5, 5)), dlabel='optimal', ilabel='optimal', skip=1)
        plot_traj_phy(ax, xs_ply, marker=True, skip=30, connect=True, dlabel=dlabel, ilabel=ilabel)

        plt.gcf().subplots_adjust(bottom=0.15, left=0.12)
        ax.set_yticks(ax.get_yticks()[::1])
        ax.grid()
        ax.axis('equal')
        plt.gca().legend(prop={'size': 12})
        
        plt.xlabel('x', fontsize=fontsize)
        plt.ylabel('y', fontsize=fontsize)
        # plt.title(fig_title, fontsize=fontsize)

        plt.savefig(file_name)
        plt.show()
        plt.close(fig)

    def _plot_value(self, v_anl, v_vfn, ts_anl, ts_vfn, fig_title, file_name, fontsize=16):

        fig, ax = plt.subplots()
        plt.rc('text', usetex=True)
        
        ax.plot(ts_anl, v_anl, linestyle=(0, (5, 5)), color='b')
        ax.plot(ts_vfn, v_vfn, linestyle=(0, ()),     color='b')
        ax.plot([np.amin(ts_vfn), np.amax(ts_vfn)], np.ones(2)*v_anl[-1], linestyle=(0, ()), color='g')
        # ax2.plot([np.amin(ts_vfn), np.amax(ts_vfn)], np.ones(2)*(2*pi - 2*gmm)*180/pi, linestyle=(0, ()), color='r')
        ax.grid()
        plt.xlabel(r'step', fontsize=fontsize)
        plt.ylabel(r'Value', fontsize=fontsize)
        plt.title(fig_title, fontsize=fontsize)
        plt.savefig(file_name)
        plt.close(fig)

    def _animate_traj(self, xs_anl, xs_ply, ts_anl, ts_ply, file_name, circ=False):
        
        O, C = phy_to_hds(xs_anl)
        xs_anl = traj_to_hds(xs_anl, O, C)
        O, C = phy_to_hds(xs_ply)
        xs_ply = traj_to_hds(xs_ply, O, C)

        # match size
        temp_t = np.interp(np.linspace(ts_anl[0], ts_anl[-1], len(ts_ply)), ts_anl, ts_anl)
        temp_x = np.zeros(np.shape(xs_ply))
        for i in range(6):
            temp_x[:, i] = np.interp(temp_t, ts_anl, xs_anl[:, i])
        ts_anl = temp_t
        xs_anl = temp_x

        # match time stamps
        if ts_anl[-1] >= ts_ply[-1]:
            times = ts_anl
            tterm = ts_ply[-1]
            for i in range(6):
                xs_ply[:,i] = np.interp(times, ts_ply, xs_ply[:,i])
            for j in range(len(times)):
                if times[j] > tterm:
                    xs_ply[j,:] = xs_ply[j-1,:]
        else:
            times = ts_ply
            tterm = ts_anl[-1]
            for i in range(6):
                xs_anl[:,i] = np.interp(times, ts_anl, xs_anl[:,i])
            for j in range(len(times)):
                if times[j] > tterm:
                    xs_anl[j,:] = xs_anl[j-1,:]

        xmin = min(np.amin(xs_anl[:,[0, 2, 4]]), np.amin(xs_ply[:,[0, 2, 4]]))
        xmax = max(np.amax(xs_anl[:,[0, 2, 4]]), np.amax(xs_ply[:,[0, 2, 4]]))
        dx = (xmax - xmin)*0.2
        ymin = min(np.amin(xs_anl[:,[1, 3, 5]]), np.amin(xs_ply[:,[1, 3, 5]]))
        ymax = max(np.amax(xs_anl[:,[1, 3, 5]]), np.amax(xs_ply[:,[1, 3, 5]]))
        dy = (ymax - ymin)*0.3
        # times = times[::-1]

        fig = plt.figure()
        ax = fig.add_subplot(111, autoscale_on=True, xlim=(xmin-dx, xmax+dx), ylim=(ymin-dy, ymax+dy))
        ax.set_aspect('equal')
        ax.grid()
        ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
        plot_traj_phy(ax, xs_anl, line_style=(0, (5, 5)), dlabel='optimal', skip=50)
        # plot_traj_phy(ax, xs_ply, line_style=(0, (5, 5)), label='Vfn', skip=50)
        plt.xlabel('x', fontsize=16)
        plt.ylabel('y', fontsize=16)

        tail = 60

        def generate_plot(ax, linestyle=(0, ()), alpha=0.5, label=None):
            D1, = ax.plot([], [], 'o', color='b', label=None)
            D2, = ax.plot([], [], 'o', color='b', label=None)
            I, = ax.plot([], [], 'o', color='r', label=None)
            D1tail, = ax.plot([], [], linewidth=2, color='b', linestyle=linestyle, label='Defender, '+label)
            D2tail, = ax.plot([], [], linewidth=2, color='b', linestyle=linestyle, label=None)
            Itail, = ax.plot([], [], linewidth=2, color='r', linestyle=linestyle, label='Defender, '+label)
            Dline, = ax.plot([], [], '--', color='b', label=None)

            D1cap = Circle((0, 0), Config.CAP_RANGE, fc='b', ec='b', alpha=alpha, label=None)
            D2cap = Circle((0, 0), Config.CAP_RANGE, fc='b', ec='b', alpha=alpha, label=None)
            ax.add_patch(D1cap)
            ax.add_patch(D2cap)

            return {
                        'D1': D1, 'D2': D2, 'I': I,
                        'D1tail': D1tail, 'D2tail': D2tail, 'Itail': Itail,
                        'Dline': Dline, 'D1cap': D1cap, 'D2cap': D2cap
                    }

        # anl_plots = generate_plot(ax, linestyle=(0, (6, 5)), alpha=0.3)
        ply_plots = generate_plot(ax, label='proposed')
        plt.gca().legend(prop={'size': 12})

        if circ:
            c1, c2 = evelope_circle()
            c1 = circ_to_hds(c1, O, C)
            c2 = circ_to_hds(c2, O, C)
            ax.plot(c1[:,0], c1[:,1], '--', color='b')
            ax.plot(c2[:,0], c2[:,1], '--', color='r')

        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes, fontsize=16)

        def init():

            def init_from_xs(plots, xs):
                plots['D1'].set_data([], [])
                plots['D2'].set_data([], [])
                plots['Dline'].set_data([], [])
                plots['D1cap'].center = (xs[0,0], xs[0,1])
                plots['D2cap'].center = (xs[0,2], xs[0,3])
                plots['I'].set_data([], [])

                plots['D1tail'].set_data([], [])
                plots['D2tail'].set_data([], [])
                plots['Itail'].set_data([], [])

                return plots['D1'], plots['D2'], plots['D1cap'], plots['D2cap'], plots['I'], plots['D1tail'], plots['D2tail'], plots['Itail'], plots['Dline']

            # objs_anl = init_from_xs(anl_plots, xs_anl)
            objs_ply = init_from_xs(ply_plots, xs_ply)

            time_text.set_text('')

            return objs_ply + tuple((time_text,))

        def animate(i):

            def animate_for_xs(plots, xs, i, ii):

                plots['D1'].set_data(xs[i,0], xs[i,1])
                plots['D2'].set_data(xs[i,2], xs[i,3])
                plots['Dline'].set_data([xs[i,0], xs[i,2]], [xs[i,1], xs[i,3]])
                plots['D1cap'].center = (xs[i,0], xs[i,1])
                plots['D2cap'].center = (xs[i,2], xs[i,3])

                plots['I'].set_data(xs[i,4], xs[i,5])

                plots['D1tail'].set_data(xs[ii:i+1,0], xs[ii:i+1,1])
                plots['D2tail'].set_data(xs[ii:i+1,2], xs[ii:i+1,3])
                plots['Itail'].set_data(xs[ii:i+1,4], xs[ii:i+1,5])

                return plots['D1'], plots['D2'], plots['D1cap'], plots['D2cap'], plots['I'], plots['D1tail'], plots['D2tail'], plots['Itail'], plots['Dline']

            i = np.clip(i-30, 0, len(xs_ply)-1)
            ii = np.clip(i-tail, 0, i)
            # objs_anl = animate_for_xs(anl_plots, xs_anl, i, ii)
            objs_ply = animate_for_xs(ply_plots, xs_ply, i, ii)
            time_text.set_text(time_template % (times[i]))

            return objs_ply + tuple((time_text, ))

        ani = animation.FuncAnimation(fig, animate, range(1, len(xs_ply)+100), interval=1, init_func=init)
        # ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
        #                                 repeat_delay=1000)
        ani.save(file_name)
        # plt.show()

    def dwin_traj(self, delta, T, gmm, t=5 , Nt=40, policies=[None, None, None], dlabel='', ilabel='', traj=False, animate=False, value=False):

        sub_dir = self.res_dir + '/' + 'gmm_%.2f'%(gmm) + '_delta_%.2f_'%(delta) + '_T_%.2f'%(T) + '/'
        p_name = ''
        for p in policies:
            if p is not None:
                p_name += str(p.__name__).split('_')[0] + '_'
            else:
                p_name += 'v_'
        name = sub_dir + p_name
        os.makedirs(sub_dir, exist_ok=True)
        fig_title = r'$\gamma=%.2f^\circ$, $\delta=%.2f^\circ$, $T=%.2f$, '%(gmm*180/pi, delta*180/pi, T)

        xs_anl, ss_anl, ds_anl, ts_anl = dwin_SS_traj(delta, T, t=t, Nt=Nt, gmm=gmm)
        xs_anl = xs_anl[::-1]
        ss_anl = ss_anl[::-1]
        ds_anl = ds_anl[::-1]
        ts_anl = np.amax(ts_anl) - ts_anl
        ts_anl = ts_anl[::-1]
        self.game.reset(xs=[xs_anl[0,:2], xs_anl[0,2:4], xs_anl[0,4:6]])
        xs_vfn, ss_vfn, ds_vfn, ts_vfn, info = self.game.advance(200, policies=policies)

        # self._plot_alpha(ss_anl)

        if traj:
            self._log_traj(xs_anl, name+'_anl.csv')
            self._log_traj(xs_vfn, name+'_ply.csv')
            self._plot_traj(xs_anl, xs_vfn, fig_title+info, name+'traj'+'.png', dlabel=dlabel, ilabel=ilabel)
        if animate:
            self._animate_traj(xs_anl, xs_vfn, ts_anl, ts_vfn, name+'traj'+'.gif')    
        
        if value:
            v_anl = self.game.Vfn.evaluate_values(ss_anl)*180/pi
            v_vfn = self.game.Vfn.evaluate_values(ss_vfn)*180/pi
            self._plot_value(v_anl, v_vfn, ts_anl, ts_vfn, fig_title+info, name+'value'+'.png')

    def iwin_traj(self, tht, delta, L=100., nT=40, length=0.5, policies=[None, None, None], dlabel='', ilabel='', animate=False):

        sub_dir = self.res_dir + '/' + 'tht_%d'%(tht*100) + '_delta_%d'%(delta*100) + '/'
        p_name = ''
        for p in policies:
            if p is not None:
                p_name += str(p.__name__).split('_')[0] + '_'
            else:
                p_name += 'v_'
        name = sub_dir + p_name
        os.makedirs(sub_dir, exist_ok=True)
        fig_title = r'$\theta=%.2f^\circ$, $\delta=%.2f^\circ$ '%(tht*180/pi, delta*180/pi)

        xs_anl, ss_anl, ds_anl, ts_anl = iwin_SS_straight_traj(tht=tht, delta=delta, L=L, nT=nT, length=length)

        self.game.reset(xs=[xs_anl[0,:2], xs_anl[0,2:4], xs_anl[0,4:6]])
        v_anl = self.game.Vfn.evaluate_value(ss_anl[0])
        xs_vfn, ss_vfn, ds_vfn, ts_vfn, info = self.game.advance(450, policies=policies)

        self._plot_traj(xs_anl, xs_vfn, fig_title+info, name+'traj'+'.png', dlabel=dlabel, ilabel=ilabel)
        self._log_traj(xs_anl, name+'_anl.csv')
        self._log_traj(xs_vfn, name+'_ply.csv')
        if animate:
            self._animate_traj(xs_anl, xs_vfn, ts_anl, ts_vfn, name+'traj'+'.gif') 

        v_anl = self.game.Vfn.evaluate_values(ss_anl)*180/pi
        v_vfn = self.game.Vfn.evaluate_values(ss_vfn)*180/pi
        self._plot_value(v_anl, v_vfn, ts_anl, ts_vfn, fig_title+info, name+'value'+'.png')
        
    #     fig3 = plt.figure()
    #     ax3 = plt.gca(projection='3d')
    #     plt.rc('text', usetex=True)
    #     plot_state_space_boundary(ax3)
    #     plot_fitted_SS(ax3, game.Vfn.evaluate_value, v_vfn[int(len(v_vfn)/3)])
    #     plot_traj_thtalpha(ax3, ss_vfn, color='b', skip=10)
    # #    plot_traj_thtalpha(ax3, ss_anl, color='k', skip=10)
    #     ax3.set_xlabel(r'$\alpha_1$', fontsize=16)
    #     ax3.set_ylabel(r'$\alpha_2$', fontsize=16)
    #     ax3.set_zlabel(r'$\theta$')
    #     plt.title(r'$\theta=%.2f^\circ$, $\delta=%.2f^\circ$ '%(tht*180/pi, delta*180/pi)+info, fontsize=16)
    #     plt.savefig(name+'_trajonSS'+'.png')
    # #    plt.show()
    #     plt.close(fig3)

    def terminal_value(self):
        
        gmms = np.linspace(LB, 0.8*pi/2, 10)
        thts = 2*pi - 2*gmms
        ds = [np.array([Config.CAP_RANGE, Config.CAP_RANGE, tht]) for tht in thts]
        
        vs_anl, vs_fit = [], []
        for d, gmm in zip(ds, gmms):
            v_anl = (2*pi - 2*gmm)*180/pi
            v_fit = self.game.Vfn.evaluate_value(d)*180/pi
            # print(v_anl, v_fit, vt_fit)
            vs_anl.append(v_anl)
            vs_fit.append(v_fit)
        
        fig, ax = plt.subplots()
        ax.plot(thts, vs_anl, color='b', marker='.')
        ax.plot(thts, vs_fit, color='r', marker='.')
        plt.grid()
        plt.show()


replay = Replay(vfunc_dir='fitted_v/', res_dir='replay')

# replay.Vcontour(pi-0.3)
# replay.conv()


# replay.terminal_value()
replay.dwin_traj(0.0999999, LB+0.4, LB/2+0.1, policies=[s_strategy]*2 + [s_strategy], dlabel='proposed', ilabel='proposed', traj=True)
# replay.dwin_traj(0.1999999, LB, LB/2+0.2, policies=[s_strategy]*2+[s_strategy], animate=True)
# replay.dwin_traj(0.1, 1, LB/2+0.2, policies=[s_strategy]*2+[s_strategy], traj=True)
# replay.iwin_traj(pi-LB+0.1, pi/2+0.2, L=100., nT=20, length=0.5, policies=[h_strategy]*2+[i_strategy], dlabel='S1', ilabel='S2')

