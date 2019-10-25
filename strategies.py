import numpy as np
from math import pi, cos, sin, atan2, sqrt, tan
from scipy.optimize import minimize_scalar

from Config import Config
r = Config.CAP_RANGE
LB = Config.LB
from coords import get_vecs, phy_to_xyz, phy_to_thtalpha, phy_to_thtd

def e_strateby(x, p):
	if abs(r-x[0]) < 0.05*r and abs(r-x[1]) < 0.05*r:
		p[0] = 0
		p[1] = 0
		p[2] = -x[2]/2
	elif abs(r-x[0]) < 0.05*r:
		p[2] = pi - x[2] - LB/2
		p[0] = 0
	elif abs(r-x[1]) < 0.05*r:
		p[2] = -(pi - LB/2)
		p[1] = 0

	return p

def t_strategy(x):
	s = phy_to_thtalpha(x)

	return np.array([-pi/10, pi/3, -s[2]/6])

def z_strategy(x):

	s = phy_to_thtd(x)

	phi_1 = -pi/2
	phi_2 = pi/2

	cpsi = s[1]*sin(s[2])
	spsi = -(s[0] - s[1]*cos(s[2]))

	psi = atan2(spsi, cpsi)

	# return e_strateby(s, np.array([phi_1, phi_2, psi]))
	return np.array([phi_1, phi_2, psi])

def h_strategy(x):

	from Config import Config
	a = Config.VD/Config.VI

	s = phy_to_xyz(x)
	x_ = np.array([0, -s[2], 0, s[2], s[0], s[1]])

	Delta = sqrt(np.maximum(s[0]**2 - (1 - 1/a**2)*(s[0]**2 + s[1]**2 - (s[2]/a)**2), 0)) 
	# print(s[0]**2 - (1 - 1/a**2)*(s[0]**2 + s[1]**2 - (s[2]/a)**2))

	if (s[0] + Delta)/(1 - 1/a**2) - s[0] > 0:
		xP = (s[0] + Delta)/(1 - 1/a**2)
	else:
		xP = - (s[0] + Delta)/(1 - 1/a**2)
	# print(Delta)
	# print(s[0], xP)
	# xP = (s[0] + Delta)/(1 - 1/a**2)
	P = np.array([xP, 0 , 0])
	D1_P = P - np.concatenate((x_[0:2], [0]))
	D2_P = P - np.concatenate((x_[2:4], [0]))
	I_P  = P - np.concatenate((x_[4:6], [0]))
	D1_I, D2_I, D1_D2 = get_vecs(x_)
	print(I_P)

	phi_1 = atan2(np.cross(D1_I, D1_P)[-1], np.dot(D1_I, D1_P))
	phi_2 = atan2(np.cross(D2_I, D2_P)[-1], np.dot(D2_I, D2_P))
	psi = atan2(np.cross(-D2_I, I_P)[-1], np.dot(-D2_I, I_P))

	# print(s, phi_1, phi_2)

	return np.array([phi_1, phi_2, psi])
	# return e_strateby(phy_to_thtd(x), np.array([phi_1, phi_2, psi]))

def s_strategy(x):

	s = phy_to_thtalpha(x)
	ds = phy_to_thtd(x)

	phi_1 = -(pi/2 - s[0])
	psi_ = s[2] - (s[0] + pi/2 - LB/2)

	d = (ds[0]/sin(LB/2))*sin(-phi_1)
	l2 = sqrt(d**2 + ds[1]**2 - 2*d*ds[1]*cos(psi_))
	s_B = (sin(psi_)/l2)*d
	c_B = (ds[1]**2 + l2**2 - d**2)/(2*ds[1]*l2)
	B = atan2(s_B, c_B)
	A = pi - B - psi_
	psi = -psi_


	def get_K(Omega):
		K = 2*pi - pi/2 - A - (pi - Omega)/2
		return K

	def delta(Omega):
		K = get_K(Omega)
		l0 = 2*r/sin(LB/2)*sin(Omega/2)
		dI = d + Omega/(tan(LB/2)/r)
		dD = sqrt(l2**2 + l0**2 - 2*l0*l2*cos(K)) - r
		err = dI*(Config.VD/Config.VI) - dD
		# err = (d*(Config.VD/Config.VI) + Omega/(tan(LB/2)/r) + r) - (l2/sin(LB/2 + Omega/2)*sin(K))
		# err = (l2/sin(LB/2 + Omega/2)*sin(K))
		return err**2

	if (s[2] - (s[0] + s[1]) <= pi - LB) and (d*(Config.VD/Config.VI) < l2 - r):
		print('curved')
		res = minimize_scalar(delta, bounds=[0.01, pi/1.5], method='bounded')
		Omega = res.x
		print(Omega)
		phi_2 = pi - get_K(Omega) - (LB/2 + Omega/2) + B
	else:
		print('straight')
		phi_2 = B

	return np.array([phi_1, phi_2, psi])
	# return e_strateby(phy_to_thtd(x), np.array([phi_1, phi_2, psi]))

def i_strategy(x):
	s = phy_to_thtalpha(x)
	ds = phy_to_thtd(x)

	phi_2 = pi/2 - s[1]
	psi = -(pi/2 - LB/2 + s[1])
	d = ds[1]*(sin(phi_2)/sin(LB/2))
	l1 = sqrt(ds[0]**2 + d**2 - 2*ds[0]*d*cos(s[2] + psi))
	
	cA = (d**2 + l1**2 - ds[0]**2)/(2*d*l1)
	sA = sin(s[2] + psi)*(ds[0]/l1)
	A = atan2(sA, cA)

	phi_1 = -(pi - (s[2] + psi) - A)
	return np.array([phi_1, phi_2, psi])
	# return e_strateby(phy_to_thtd(x), np.array([phi_1, phi_2, psi]))

def m_strategy(x):
	s = phy_to_xyz(x)
	if s[0]<-0.8*Config.VI:
		print('h')
		return h_strategy(x)
	else:
		return i_strategy(x)