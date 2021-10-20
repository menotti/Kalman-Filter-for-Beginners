import numpy as np 
from math import atan2, asin

class KalmanFilter(object):
	def __init__(self):
		#initial Variables
		self.A = np.matrix
		self.H = np.matrix
		self.Q = np.matrix
		self.R = np.matrix
		self.x = np.matrix
		self.P = np.matrix
		self.dt : float
		self.z = np.matrix
		self.K = np.matrix

	def setTransitionMatrix(self, A):
		self.A = A
        
	def setSttMeasure(self, H):
		self.H = H

	def setSttVariable(self, x):
		self.x = x

	def setTransitionCovMatrix(self, Q):
		self.Q = Q

	def setMeasureCovMatrix(self, R):
		self.R = R

	def setDeltaT(self, delta):
		self.dt = delta

	def setErrorCovMatrix(self, P):
		self.P = P

	def filter(self, A, z):
		self.z = z
		xp = A * self.x
		Pp = A * self.P * A.getT() + self.Q
		self.K = Pp * self.H.getT() * np.matrix.getI(self.H * Pp * self.H.getT() + self.R)
		self.x = xp + self.K * (self.z - self.H * xp)
		self.P = Pp - self.K * self.H * Pp

		x1 = self.x[0, 0]
		x2 = self.x[1, 0]
		x3 = self.x[2, 0]
		x4 = self.x[3, 0]

		phi   =  atan2(2*(x3*x4 + x1*x2), 1 - 2*(x2**2+x3**2))
		theta = -asin( 2*(x2*x4 - x1*x3))
		psi   =  atan2(2*(x2*x3 + x1*x4), 1 - 2*(x3**2+x4**2))

		return phi, theta, psi
