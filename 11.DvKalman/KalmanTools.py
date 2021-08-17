import numpy as np

def GetSonar():
    """
    This function simulates as an sonar, sending back a list of
    values and its total samples that comes from GetSonar.csv
    :return:
    1)measures: An array list related to salved values in GetSonar.csv
    2)Samples: The total values obtained by the file
    """
    file_name = 'GetSonar.csv'
    file = open(file_name)
    data = file.read()
    measures = data.split(';')
    Samples = len(measures)
    file.close()
    measures = [float(x) for x in measures]
    return measures, Samples

def GetPos(dt=0.1):
    """
    This function returns velocity and position as output.
    Although, it's return contains noises, like a sensor.
    :param dt: (optional) instantiate at start of the program
    :return: measured position of the object
    """
    posp = 0
    velp = 80
    i = 0
    while True:
        v = 0 + 10 * np.random.randn()
        w = 0 + 10 * np.random.randn()
        i += 1
        value = posp + velp * dt + v
        posp = value - v  # true position
        velp = 80 + w  # true speed
        yield value

class KalmanFilter:
    # You should avoid to use at the same object the following 
    # functions: dv_filter and de_filter
    def __init__(self):
        self.dt = 0.1
        self.A = np.matrix([[1., self.dt],
                            [0., 1.]])

        self.H = np.matrix([1., 0.])

        self.Q = np.matrix([[1., 0.],
                            [0., 3.]])

        self.R = np.matrix([10.])
        
        self.x = np.matrix([0., 20.]).getT()
        self.P = 5 * np.identity(2)
        self.dv_z: float
        self.de_z: float
        self.K = np.zeros((2, 1))

    def dv_filter(self, dv_position):
        self.dv_z = dv_position
        xp = self.A * self.x
        Pp = self.A * self.P * self.A.getT() + self.Q
        self.K = Pp * self.H.getT() * np.matrix.getI(self.H * Pp * self.H.getT() + self.R)
        self.x = xp + self.K * (self.dv_z - self.H * xp)
        self.P = Pp - self.K * self.H * Pp
        dv_pos = self.x[0, 0]
        dv_vel = self.x[1, 0]
        return dv_pos, dv_vel

    def de_filter(self, de_position):
        self.de_z = de_position
        xp = self.A * self.x
        Pp = self.A * self.P * self.A.getT() + self.Q
        self.K = 1 / (Pp[0, 0] + self.R) * np.matrix([Pp[0, 0], Pp[1, 0]])
        # (Comment the next one)
        self.x = xp + self.K * (self.de_z - self.H * xp)
        self.P = Pp - self.K * self.H * Pp
        de_pos = self.x[0, 0]
        de_vel = self.x[1, 0]
        return de_pos, de_vel
