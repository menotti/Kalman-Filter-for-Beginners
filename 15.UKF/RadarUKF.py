#libraries used in this example
import numpy as np
from scipy.linalg import inv, cholesky



class UKF(object):
	"""docstring for UKF"""
	def __init__(self):

		self.x = np.array([0,90,1100])
		self.m = self.x.size
		self.n = 1
		self.Q = 0.01 * np.eye(self.m)  # Process noise matrix
		self.R = np.array([100]) # Measurement noise matrix
		self.P = 100*np.eye(self.m)		

		self.kappa = 0
		self.dt = 0


		# Array com Pesos dos Sigma Points
		self.W = self.calc_weight()


	def calc_weight(self):
		size = self.m*2+1
		W = np.full(size, 1./(2*(self.m+self.kappa)))
		W[0] = self.kappa / (self.m + self.kappa)

		return W

	#DONE 
	def SigmaPoints(self): 
		n = self.m
		x = self.x
		U = cholesky((n+self.kappa)*self.P)
		Xi = np.zeros((2*self.m+1, self.m))

		Xi[0] = x
		for k in range(n):
			Xi[k+1]   = x + U[k]

			Xi[n+k+1] = x - U[k]
		test = np.zeros((2*self.m+1, self.m))
		for i in range(n):
			test[0,i] = x[i]

		# parallel for 
		for i in range(n): 
			# soma dos elementos em U + x
			for k in range(n): 

				test[i+1,k]   = x[k] + U[k,i] 

				test[n+i+1,k] = x[k] - U[k,i]  

		return Xi

	
	def UT(self, sigma, Noise, W = None):
		
		if W == None:
			W = self.W
		xm = np.dot(W, sigma)
	
		

		size, n = sigma.shape
		cov = np.zeros((n,n))
		
		aux = np.zeros((n, n))
		

		for i in range(size):
			Y = sigma[i] - xm #axpy_batch
			Y = Y.reshape(n, 1)


			aux = W[i] * np.dot(Y, Y.T)

			cov += aux#gemm_batch			

		cov += Noise
		return xm, cov
		

	def hx(self,x):
		return (x[0]**2 + x[2]**2)**(1/2)

	def UT_Hx(self, sigma):

		size, _ = sigma.shape

		Hxi_pts = np.zeros((size, 1))
		for i in range(size):
			Hxi_pts[i] = self.hx(sigma[i])
	
		return Hxi_pts


	def fx(self, x, dt):
		A = np.eye(3) + dt*np.array([[0, 1, 0],
								      [0, 0, 0],
								      [0, 0, 0]])

		return np.matmul(A, x)

	def UT_Fx(self, Xi, dt):
		size, _ = Xi.shape

		Fxi_pts = np.zeros((size, self.m))
		
		for i in range(size):

			Xi_vector = np.array(Xi[i]).T
			
			Fxi_pts[i] = self.fx(Xi_vector, dt)
	
		return Fxi_pts


	def UKF(self, z, dt):
		
		Xi  = self.SigmaPoints()
		fXi = self.UT_Fx(Xi, dt)
		
		hXi = self.UT_Hx(Xi)
			
		xp, Pp = self.UT(fXi, self.Q)
		zp, Pz = self.UT(hXi, self.R)

		# ===/= Estou aqui (oneAPI)=/===
		
		Pxz = np.zeros((self.m,self.n))

		for i in range(2*self.m+1): #gemm_batch
			a = fXi[i] - xp
			b = hXi[i] - zp
			Pxz+= self.W[i]*np.outer(a, b)		
		

		K = np.dot(Pxz, inv(Pz))
		self.x = xp + np.dot(K, (z-zp))
		

		self.P = Pp - np.dot(np.dot(K, Pz),K.T)
		Kpz = np.dot(K,Pz)
		P1 = np.dot(Kpz, K.T)
		P  = Pp - P1
		pos = self.x[0]
		vel = self.x[1]
		alt = self.x[2]
		


		return pos, vel, alt


