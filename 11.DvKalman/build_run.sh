#!/bin/bash
source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
make
# if [ $? -eq 0 ]; then ./DvKalman; fi
make clean
