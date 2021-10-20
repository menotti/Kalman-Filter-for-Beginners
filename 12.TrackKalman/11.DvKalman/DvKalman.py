# Libraries used in this code
import numpy as np
import matplotlib.pyplot as plot
import KalmanTools as Tools
np.random.seed(42)


# instantiate generator and lists signals for plot
dt = 0.1
t = np.arange(0, 10, dt)
NSamples = len(t)
position = Tools.GetPos()
XSaved = np.zeros((2, NSamples))
ZSaved = np.zeros(NSamples)

Kalman = Tools.KalmanFilter()
Kalman.dt = dt
for k in range(NSamples):
    z = ZSaved[k] = next(position)
    XSaved[0:2, k] = Kalman.dv_filter(z)


first_plot, pos_plot = plot.subplots()
second_plot, vel_plot = plot.subplots()
pos_plot.plot(t, XSaved[0], 'b-', label='Kalman Filter')
pos_plot.plot(t, ZSaved, 'ro--', label='Measure', markersize=2)
pos_plot.set(title="Position Label", xlabel="Time [sec]", ylabel="Position [Meters]")
leg_1 = pos_plot.legend()


vel_plot.plot(t, XSaved[1], 'o-', label='Velocity', markersize=1)
vel_plot.set(title="Velocity Label", xlabel="Time [sec]", ylabel="Velocity [M/s]")
leg_2 = vel_plot.legend()


