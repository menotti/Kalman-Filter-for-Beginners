# Makefile for GNU Make

default: run

all: run

run: EulerKalman
	./EulerKalman

MKL_COPTS = -DMKL_ILP64  -I"${MKLROOT}/include"
MKL_LIBS = -L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl

DPCPP_OPTS = $(MKL_COPTS) -fsycl-device-code-split=per_kernel $(MKL_LIBS)

EulerKalman: EulerKalman.cpp
	dpcpp $< -o $@ $(DPCPP_OPTS)

clean:
	-rm -f EulerKalman

.PHONY: clean run all

