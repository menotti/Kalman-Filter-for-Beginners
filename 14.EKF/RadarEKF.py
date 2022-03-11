
import numpy as np

# Jacobian Function Declaration
def Hjacob(xhat):

	x1 = xhat[0, 0]
	x3 = xhat[2, 0]

	H = np.matrix([[0., 0., 0.]])
	H[0, 0] = x1/(x1**2+x3**2)**(1/2)
	H[0, 1] = 0
	H[0, 2] = x3/(x1**2+x3**2)**(1/2)
	
	return H


def hx(xhat):
	x1 = xhat[0, 0]
	x3 = xhat[2, 0]
	zp = np.matrix([[(x1**2 + x3**2)**(1/2)]])

	return zp

class RadarEKF:
	def __init__(self, dt):
        # Declare Member-functions 
		self.A  = np.eye(3) + dt*np.matrix([[0, 1., 0],
											[0, 0 , 0],
											[0, 0 , 0]])
        
        #Measurement Noise Matrix
		self.Q = np.matrix([[0, 0    , 0    ],
							[0, 0.001, 0    ],
							[0, 0    , 0.001]])
        

		self.R = np.matrix([[10]])
        
        # State Matrix
		self.x = np.matrix([[0.0, 90.0, 1100.0]]).getT()
    
        #Measurement (?) Matrix
		self.P = np.eye(3) * 10


	def measure(self, z):
		H  = Hjacob(self.x)
		xp = self.A*self.x
		Pp = self.A * self.P * self.A.getT() + self.Q
		
		K = Pp * H.getT() * np.matrix.getI(H * Pp * H.getT() + self.R)
		
		self.x = xp + K * np.matrix(z - hx(xp))

		self.P = Pp - K * H * Pp

		pos = self.x[0, 0]
		vel = self.x[1, 0]
		alt = self.x[2, 0]
		return pos, vel, alt



