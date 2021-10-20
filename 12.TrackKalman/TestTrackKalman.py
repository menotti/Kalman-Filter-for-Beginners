import numpy as np
import matplotlib.pyplot as plot
import KalmanTools as Tools

def GetBallPos():
    filename = "Measure_Img.csv"
    with open(filename) as file_obj:
        lines = file_obj.readlines()
        X = lines[0].split(",")
        Y = lines[1].split(",")
    X = [float(x) for x in X]
    Y = [float(x) for x in Y]
    i = 0
    while i < len(X):
        x =X[i]
        y =Y[i]
        yield x,y
        i +=1



NoOfImg = 24
dt = 1

Xhsaved = np.zeros((2,NoOfImg))
Xmsaved = np.zeros((2,NoOfImg))
XmRead = GetBallPos()

TrackKalman = Tools.KalmanFilter()

A = np.matrix([[1,dt, 0, 0],
		       [0, 1, 0, 0],
		       [0, 0, 1,dt],
		       [0, 0, 0, 1]])

H = np.matrix([[1, 0, 0, 0],
               [0, 0, 1, 0]])

Q = np.eye(4)
R = np.eye(2)*50
x = np.zeros((4,1))
P = np.eye(4)*100

TrackKalman.setDeltaT(dt)
TrackKalman.setTransitionMatrix(A)
TrackKalman.setSttMeasure(H)
TrackKalman.setTransitionCovMatrix(Q)
TrackKalman.setErrorCovMatrix(P)
TrackKalman.setMeasureCovMatrix(R)
TrackKalman.setSttVariable(x)


for i in range(NoOfImg):
	xm, ym = Xmsaved[0:2, i] = next(XmRead)
	result = TrackKalman.filter(xm, ym)
	Xhsaved[0:2, i] = result


# plotting results
xy, position = plot.subplots()


position.plot(Xmsaved[0], Xmsaved[1], '*', label='Measured')
position.plot(Xhsaved[0], Xhsaved[1], 's', label='Kalman Filter')
position.set(title="Position Label", xlabel="Horizontal [pixel]", ylabel="Vertical [pixel]")
position.set_ylim([0,250])
position.set_xlim([0,350])
leg_1 = position.legend()


plot.show()
