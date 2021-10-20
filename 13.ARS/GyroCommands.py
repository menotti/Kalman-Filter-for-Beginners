import csv
from math import cos, sin, tan, pi
import matplotlib.pyplot as plot
import numpy as np


class Gyro:
    def __init__(self):
        self.prev_angles = [0, 0, 0]
        self.phi = 0
        self.theta = 0
        self.psi = 0

    def GetGyro(self, filename):
        with open(filename) as file:
            data = csv.reader(file)
            for line in data:
                line = [float(x) for x in line]
                yield line

    def EulerGyro(self, R, delta):
        p_phi, p_theta, p_psi = self.prev_angles

        phi = p_phi     + delta * (R[0] + R[1]*sin(p_phi) * tan(p_theta) + R[2] * cos(p_phi) * tan(p_theta))
        theta = p_theta + delta * (       R[1]*cos(p_phi)                  - R[2] * sin(p_phi))
        psi = p_psi     + delta * (       R[1]*sin(p_phi)/cos(p_theta)     + R[2] * cos(p_phi) / cos(p_theta))
        self.prev_angles = phi,theta,psi
        return [phi, theta, psi]

