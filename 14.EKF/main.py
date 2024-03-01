import numpy as np
import RadarEKF as rf
import matplotlib.pyplot as plt
import os
import sys
import time

def GetRadar(dt):
    posp = 0
    while True:
        # Estipulate velocity and altitude
        vel = 100  +  5*np.random.randn()
        alt = 1000 + 10*np.random.randn()
        # Refresh position
        pos = posp + vel * dt
        # Read velocity
        v = pos * 0.05 * np.random.randn()
        r = (pos**2 + alt**2)**(0.5) + v
    
        posp = pos
        yield r

def main(argv):
    # Declare Initial Variables
    kIterations = 1
    if(len(sys.argv) == 2):
        kIterations = int(sys.argv[1])

    print("Running Samples with {} Iteration".format(kIterations))
    dt = 0.05
    t = np.arange(0,20,dt)
    Nsamples = len(t)
    Xsaved = np.zeros((3, Nsamples))
    Zsaved = np.zeros((1, Nsamples))


    # Instantiate Obj. used in this example
    measure = GetRadar(dt)
    EKF = rf.RadarEKF(dt)


    # Foor  loop -> measure 
    beg = time.time()
    for iter in range(kIterations):
        for k in range(Nsamples):
            # Get position
            r = next(measure)
            
            # Filter usage
            pos, vel, alt = EKF.measure(r)
            Xsaved[:, k] = pos, vel, alt
            
            # Store values
            Zsaved[0, k] = r
            # detach Xsaved --> Pos, Vel, Alt

    total = time.time() - beg
    PosSaved = Xsaved[0, :]
    VelSaved = Xsaved[1, :]
    AltSaved = Xsaved[2, :]

    print("Elapsed Time {:.4f} [ms]".format(total/kIterations*1e3))
    # Plot results
    plot, dx = plt.subplots()
    plot, dv = plt.subplots()
    plot, ax = plt.subplots()

    dx.plot(t, PosSaved, '-', label="Position", markersize=0.5)
    dv.plot(t, VelSaved, '-', label="Velocity", markersize=0.5)
    ax.plot(t, AltSaved, '-', label="Altitude", markersize=0.5)

    # plt.show()


main(sys.argv)