#include <iostream>

#include <vector>
using std::vector; 

#include <fstream>
using std::ifstream; 
using std::ofstream; 


#include <stdlib.h>


#include <string>
using std::stof;
using std::string;

#include <limits>
#include <random>

#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"
namespace blas = oneapi::mkl::blas;
namespace lapack = oneapi::mkl::lapack;

#include "RadarUKF.h"



//Function Prototype
double *GetRadar(int Nsamples,double dt, int seed);



int main(){
    //create file : TestRadarUKF.csv
    ofstream myfile("TestRadarUKF.csv");
    myfile << "time,Position,Velocity,Altitude\n";
    auto async_handler = [](sycl::exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                rethrow_exception(e);
            }
            catch (sycl::exception const &e) {
                std::cout << "Caught asynchronous SYCL exception: " << e.what() << std::endl;
            }
        }
    };
    
    try {
        
         
        //About Accelerator Device & Queue
        sycl::device device = sycl::device(sycl::default_selector());
        std::cout << "Device: " << device.get_info<sycl::info::device::name>() << "\n";
        sycl::queue queue(device, async_handler);
        vector<sycl::event> event_list;
        
        
        
        //Declare example variables, lists and Kalman Object
        double dt = 0.05;
        
        
        const int Nsamples = 400;
        double t[Nsamples];
        const int seed = 777; 
        double *Radius = GetRadar(Nsamples, dt, seed);
          
        //#set initial matrix:
        auto *Q = sycl::malloc_shared<double>(3 * 3, queue);
        eye(Q, 3, 0.01); 
        
        auto *R = sycl::malloc_shared<double>(1 * 1, queue);
        R[0] = 100.0;
        
        auto *x = sycl::malloc_shared<double>(3 * 1, queue);
        x[0] = 0.0; x[1] = 90.0; x[2]  = 1100.0; 
        
        auto *P = sycl::malloc_shared<double>(3 * 3, queue);
        eye(P, 3, 100.0);
    
        auto *z = sycl::malloc_shared<double>(1 * 1, queue);
        
        // #Set RadarUKF objects 
        
        RadarUKF *Rf = new RadarUKF();
        Rf->setTransitionCovMatrix(Q);
        Rf->setMeasureCovMatrix(R);
        Rf->setErrorCovMatrix(P);
        Rf->setSttVariable(x);
        
        auto *PosSaved = sycl::malloc_shared<double>(Nsamples, queue);
        auto *VelSaved = sycl::malloc_shared<double>(Nsamples, queue);
        auto *AltSaved = sycl::malloc_shared<double>(Nsamples, queue);
        
        if (!PosSaved || !VelSaved || !AltSaved) {
            std::cerr << "Could not allocate memory for vectors." << std::endl;
            exit(1);
        }
        
        double pos, alt, vel; 
        for (int i = 0; i < Nsamples ; i++){  
                
            z[0] = Radius[i];

            t[i] = dt*i;
            
            
            
            //Update then save.
            
            Rf->UKF(z, dt);
            pos = Rf->getResult(0);
            alt = Rf->getResult(1);
            vel = Rf->getResult(2);
            
            
            //#then store on arrays
            
            PosSaved[i] = pos; 
            AltSaved[i] = alt; 
            VelSaved[i] = vel; 
            
            
            //#Save the results below in myfile. 
            myfile << t[i] <<", "<< pos << ", " << vel << ", " << alt <<'\n';
            
        }
        myfile.close();
        free(PosSaved, queue);
        free(VelSaved, queue);
        free(AltSaved, queue);
        
        //kalman->end_task();

    } catch (const exception &e) {
        std::cerr << "An exception occurred: "
                  << e.what() << std::endl;
        exit(1);
    }
}

double *GetRadar(int Nsamples,double dt, int seed){
	int posp = 0;
	int i = 0; 

	std::default_random_engine generator;
	generator.seed(seed);
	std::normal_distribution<double> distribution(0.0, 1.0);

	double *radar = new double[Nsamples];

	double vel, v, r, alt, pos;
	while (i < Nsamples){ 
        
		vel = 100  +  5*distribution(generator);
        alt = 1000 + 10*distribution(generator);
        pos = posp + vel*dt;

        v = pos *.05 * distribution(generator);
        r = sqrt(pos*pos + alt*alt) +v;
        radar[i++] = r;
        posp = pos;
    
	}

	return radar;
}


