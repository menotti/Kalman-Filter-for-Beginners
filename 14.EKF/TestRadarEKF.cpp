#include <iostream>
    using std::cout;
using std::endl;

#include <vector>
using std::vector; 

#include <fstream>
using std::ifstream; 
using std::ofstream; 


#include <stdlib.h>
#include <string>
using std::stof;
using std::string;

#include <chrono>
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

#include <limits>
#include <random>
#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"

#include "RadarEKF.h"

namespace blas = oneapi::mkl::blas;
namespace lapack = oneapi::mkl::lapack;


//Function Prototypes
DATA_TYPE *GetRadar(int Nsamples,DATA_TYPE dt);
void eye(DATA_TYPE* A, int l, DATA_TYPE a );
void zero(DATA_TYPE* A, int l, int m);
void display(DATA_TYPE *A, int nRows, int nCols);

int main(int argc, char **argv){
    //create file : EulerKalman.csv
    auto async_handler = [](sycl::exception_list exceptions) {
        for (exception_ptr const &e : exceptions) {
            try {
                rethrow_exception(e);
            }
            catch (sycl::exception const &e) {
                cout << "Caught asynchronous SYCL exception: " << e.what() << endl;
            }
        }
    };
    
    try {
        int kIterations =1;
        if(argc == 2)
            kIterations = atoi(argv[1]);
        
        //About Accelerator Device & Queue
#if defined(_SYCL_GPU)
        sycl::device device = sycl::device(sycl::gpu_selector_v);
        std::string file_result = "RadarEKF_gpu.csv";
#else
        sycl::device device = sycl::device(sycl::cpu_selector_v);
        std::string file_result = "RadarEKF_cpu.csv";
#endif
        sycl::queue queue(device, async_handler);
        cout << "Device:    \t " << device.get_info<sycl::info::device::name>() << "\n";
        cout << "Number Iterations: " << kIterations << "\n";

        vector<sycl::event> event_list;
        
        
        
        //Declare example variables, lists and Kalman Object
        DATA_TYPE dt = 0.05;
        const int Nsamples = 400;
        DATA_TYPE t[Nsamples];
        
        DATA_TYPE *Radius = GetRadar(Nsamples, dt);
        

        //Declare Default Matrix, in accord to the Example; 
        auto *A = sycl::malloc_shared<DATA_TYPE>(3 * 3, queue);
        eye(A,3,1.0); A[1] = dt;
                
        auto *Q = sycl::malloc_shared<DATA_TYPE>(3 * 3, queue);
        eye(Q, 3, 0.0001); Q[0] = 0.0; 
        
        auto *R = sycl::malloc_shared<DATA_TYPE>(1 * 1, queue);
        R[0] = 10.0;
        
        auto *x = sycl::malloc_shared<DATA_TYPE>(3 * 1, queue);
        x[0] = 0.0; x[1] = 90; x[2]  = 1100.0;   
        
        auto *P = sycl::malloc_shared<DATA_TYPE>(3 * 3, queue);
        eye(P, 3, 10.0);
        
        auto *z = sycl::malloc_shared<DATA_TYPE>(1 * 1, queue);
        
        auto *PosSaved = sycl::malloc_shared<DATA_TYPE>(Nsamples, queue);
        auto *VelSaved = sycl::malloc_shared<DATA_TYPE>(Nsamples, queue);
        auto *AltSaved = sycl::malloc_shared<DATA_TYPE>(Nsamples, queue);
        
        
        RadarEKF EKF(queue, 3, 1);
        
        EKF.setTransitionMatrix(A);
        EKF.setTransitionCovMatrix(Q);
        EKF.setMeasureCovMatrix(R);
        
        EKF.setErrorCovMatrix(P);
        EKF.setSttVariable(x);
        
        
        if (!PosSaved || !VelSaved || !AltSaved) {
            cerr << "Could not allocate memory for vectors." << endl;
            exit(1);
        }
        
        DATA_TYPE pos, alt, vel; 

        auto beg = high_resolution_clock::now();
        for(int iter=0; iter< kIterations; iter++)
            for (int i = 0; i < Nsamples; i++){  
                //Build up Time array
                t[i] = i*dt;
                //Get Prev. value stored from radar
                z[0] = Radius[i];
                                    
                EKF.update(z);

                PosSaved[i] = pos = EKF.getResult(0);
                AltSaved[i] = alt = EKF.getResult(1);
                VelSaved[i] = vel = EKF.getResult(2);

                //calculate, then store on arrays
                
            }
        auto end = high_resolution_clock::now();
        
        cout << "Average time execution(ms): " << setw(2) << (double) duration_cast<milliseconds>(end - beg).count()/kIterations << "\n"; 
        
        // Save the results below in myfile. 
        ofstream myfile(file_result);
        myfile << "time,PosSaved,VelSaved,AltSaved\n";
        for(int idx = 0; idx < Nsamples; idx++)
                myfile << t[idx] <<", "<< PosSaved[idx] << ", " << VelSaved[idx] << ", " << AltSaved[idx] <<'\n';
        myfile.close();

        free(PosSaved, queue);
        free(VelSaved, queue);
        free(AltSaved, queue);
        
        //EKF.end_task();
    } catch (const exception &e) {
        cerr << "An exception occurred: "
             << e.what() << endl;
        exit(1);
    }
}



DATA_TYPE *GetRadar(int Nsamples,DATA_TYPE dt){
	int posp = 0;
	int i = 0; 

	default_random_engine generator;
	normal_distribution<float> distribution(0.0, 1.0);

	DATA_TYPE *radar = new DATA_TYPE[Nsamples];

	DATA_TYPE vel, v, r, alt, pos;
	while (i < Nsamples){ 
		vel = 100  +  0.5*distribution(generator);
        alt = 1000 + 1.0*distribution(generator);
        pos = posp + vel*dt;

        v = pos *.05 * distribution(generator);
        r = sqrt(pos*pos + alt*alt) +v;
        radar[i] = r;
        i++;
        posp = pos;
	}

	return radar;
}

void eye(DATA_TYPE* A, int l, DATA_TYPE val){
	for(int i =0; i < l; i++){
		for(int j=0; j<l; j++){
			if(i == j) A[j+i*l] = val;
			else A[j+i*l] = 0.0;
		}
	}
}
void zero(DATA_TYPE* A, int l, int m){
	for(int i =0; i < l; i++){
		for(int j=0; j<m; j++){
			A[j+i*l] = 0.0;
		}
	}
}
