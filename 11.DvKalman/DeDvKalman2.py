# Libraries used in this code
import numpy as np
import matplotlib.pyplot as plot
import KalmanTools as Tools

# generate and lists signals for plot
ZSaved, NSamples = Tools.GetSonar()
XSaved = np.zeros((2, NSamples))
dt = 0.02
t = np.arange(0, 10, dt)

# define class 'Kalman' to use it's internal modules
DvKalman = Tools.KalmanFilter()
DeKalman = Tools.KalmanFilter()
DeKalman.dt = DvKalman.dt = dt
# iteration to generate XSaved list
for k in range(NSamples):
    A = DvKalman.dv_filter(ZSaved[k])
    XSaved[0:2, k] = A

# plotting results
plot_1, position = plot.subplots()
plot_2, velocity = plot.subplots()

position.plot(t, XSaved[0], 'o-', label='Kalman Filter', markersize=1)
position.plot(t, ZSaved, 'r:*', label='Measurements', markersize=2)

position.set(title="Position Label", xlabel="Time [sec]", ylabel="Position [Meters]")
leg_1 = position.legend()


velocity.plot(t, XSaved[1], 'o-', label='Velocity', markersize=1)
velocity.set(title="Velocity Label", xlabel="Time [sec]", ylabel="Velocity [M/s]")
leg_2 = velocity.legend()
plot.show()
