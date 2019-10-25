import numpy as np
from math import pi
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

from Config import Config
from hhtrajectory import evelope_circle
from coords import phy_to_hds, traj_to_hds, circ_to_hds
 

def animate_traj(xs, times, circ=True):
	
	O, C = phy_to_hds(xs)
	xs = traj_to_hds(xs, O, C)

	xmin = np.amin(xs[:,[0, 2, 4]])
	xmax = np.amax(xs[:,[0, 2, 4]])
	dx = (xmax - xmin)*0.2
	ymin = np.amin(xs[:,[1, 3, 5]])
	ymax = np.amax(xs[:,[1, 3, 5]])
	dy = (ymax - ymin)*0.3
	# times = times[::-1]

	fig = plt.figure()
	ax = fig.add_subplot(111, autoscale_on=True, xlim=(xmin-dx, xmax+dx), ylim=(ymin-dy, ymax+dy))
	# ax = fig.add_subplot(111, autoscale_on=True, xlim=(-30, 30), ylim=(-20, 20))
	# ax = fig.add_subplot()
	ax.set_aspect('equal')
	ax.grid()

	x = xs[0]
	t = times[0]

	tail = 30
	D1, = ax.plot([], [], 'o', color='b')
	D2, = ax.plot([], [], 'o', color='b')
	I, = ax.plot([], [], 'o', color='r')
	D1tail, = ax.plot([], [], linewidth=2, color='b')
	D2tail, = ax.plot([], [], linewidth=2, color='b')
	Itail, = ax.plot([], [], linewidth=2, color='r')
	Dline, = ax.plot([], [], '--', color='b')

	D1cap = Circle((0, 0), Config.CAP_RANGE, fc='b', ec='b', alpha=0.5)
	D2cap = Circle((0, 0), Config.CAP_RANGE, fc='b', ec='b', alpha=0.5)

	ax.add_patch(D1cap)
	ax.add_patch(D2cap)

	if circ:
		c1, c2 = evelope_circle()
		c1 = circ_to_hds(c1, O, C)
		c2 = circ_to_hds(c2, O, C)
		ax.plot(c1[:,0], c1[:,1], '--', color='b')
		ax.plot(c2[:,0], c2[:,1], '--', color='r')


	time_template = 'time = %.1fs'
	time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


	def init():

	    D1.set_data([], [])
	    D2.set_data([], [])
	    Dline.set_data([], [])
	    D1cap.center = (xs[0,0], xs[0,1])
	    D2cap.center = (xs[0,2], xs[0,3])

	    I.set_data([], [])
	    time_text.set_text('')

	    return D1, D2, D1cap, D2cap, I, D1tail, D2tail, Itail, Dline, time_text

	def animate(i):
		i = np.clip(i-30, 0, len(xs)-1)

		ii = np.clip(i-tail, 0, i)

		D1.set_data(xs[i,0], xs[i,1])
		D2.set_data(xs[i,2], xs[i,3])
		Dline.set_data([xs[i,0], xs[i,2]], [xs[i,1], xs[i,3]])
		D1cap.center = (xs[i,0], xs[i,1])
		D2cap.center = (xs[i,2], xs[i,3])

		I.set_data(xs[i,4], xs[i,5])

		D1tail.set_data(xs[ii:i+1,0], xs[ii:i+1,1])
		D2tail.set_data(xs[ii:i+1,2], xs[ii:i+1,3])
		Itail.set_data(xs[ii:i+1,4], xs[ii:i+1,5])

		time_text.set_text(time_template % (times[i]))

		return D1, D2, D1cap, D2cap, I, D1tail, D2tail, Itail, Dline, time_text


	ani = animation.FuncAnimation(fig, animate, range(1, len(xs)+100), interval=30, init_func=init)
	# ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
	#                                 repeat_delay=1000)
	ani.save('hh.gif')
	plt.show()

