# Makefile for GNU Make
all: Radar_gpu Radar

default: run


run: TestRadarEKF
	./TestRadarEKF

MKL_COPTS = -DMKL_ILP64  -I"${MKLROOT}/include"
MKL_LIBS = -L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl

DPCPP_OPTS = $(MKL_COPTS) -fsycl-device-code-split=per_kernel $(MKL_LIBS)


Radar_gpu: TestRadarEKF.cpp
	icpx -fsycl -D_SYCL_GPU $< -o $@ $(DPCPP_OPTS) 

Radar: TestRadarEKF.cpp
	icpx -fsycl $< -o $@ $(DPCPP_OPTS) 

Profile_gpu:
	advisor --collect=roofline --profile-gpu --project-dir=./advisor/gpu -- ./Radar_gpu 4

Profile:
	advisor --collect=roofline --project-dir=./advisor/cpu -- ./Radar 4

clean:
	-rm -f TestRadarEKF

.PHONY: clean run all

