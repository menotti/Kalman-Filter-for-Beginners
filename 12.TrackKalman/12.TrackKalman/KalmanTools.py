import numpy as np 

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
		self.z : float
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

	def filter(self, dx, dy):
		self.z = np.matrix([[dx], [dy]])
		xp = self.A * self.x
		Pp = self.A * self.P * self.A.getT() + self.Q
		self.K = Pp * self.H.getT() * np.matrix.getI(self.H * Pp * self.H.getT() + self.R)
		self.x = xp + self.K * (self.z - self.H * xp)
		self.P = Pp - self.K * self.H * Pp
		dv_pos = self.x[0, 0]
		dv_vel = self.x[2, 0]
		return dv_pos, dv_vel
