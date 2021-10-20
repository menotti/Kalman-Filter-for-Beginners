#include <iostream>
using std::cout;
using std::endl;
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
#include "KalmanTools.h"

//Function Prototype
double *Getsonar();

int main(){
    //create file : DvKalman
    ofstream myfile("DvKalman.csv");
    myfile << "Time,Measurements,Position,Velocity\n";
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
        //Define pseudo-random values and normal distribution
        default_random_engine generator;
        generator.seed(42);
        normal_distribution<double> distribution(0.0, 1.0);
         
        //About Accelerator Device & Queue
        sycl::device device = sycl::device(sycl::default_selector());
        cout << "Device: " << device.get_info<sycl::info::device::name>() << "\n";
        sycl::queue queue(device, async_handler);
        vector<sycl::event> event_list;
        
        //Declare example variables, lists and Kalman Object
        double dt = 0.02;
        const int Nsamples = 500;
        double t[Nsamples];
        Kalman_filter *kalman = new Kalman_filter();
        kalman->default_values(dt);
        auto *Zsaved = sycl::malloc_shared<double>(Nsamples, queue);
        auto *PosSaved = sycl::malloc_shared<double>(Nsamples, queue);
        auto *VelSaved = sycl::malloc_shared<double>(Nsamples, queue);
        
        //generate list with Sonar measures
        Zsaved = Getsonar();
        
        if (!PosSaved || !VelSaved) {
            cerr << "Could not allocate memory for vectors." << endl;
            exit(1);
        }
        int first = 0;
        for (int i = 0; i < Nsamples; i++){
            t[i] = dt*i;
            //calculate, then store.
            kalman->filter(Zsaved[i]);
            PosSaved[i] = kalman->getResult(0,0);
            VelSaved[i] = kalman->getResult(0,1);
            myfile << t[i] << ", " << Zsaved[i] << ", " << PosSaved[i] << ", " << VelSaved[i] <<'\n';
        }
        myfile.close();
        free(PosSaved, queue);
        free(VelSaved, queue);
        free(Zsaved, queue);
        

    } catch (const exception &e) {
        cerr << "An exception occurred: "
                  << e.what() << endl;
        exit(1);
    }
}

double *Getsonar() {
	string Mytext;
	int i= 0, NSamples = 500;
	double *arr = new double[500];
	//Leitura de arquivo
    ifstream MyFile("GetSonar.csv");
 while(getline(MyFile, Mytext,';')){
	arr[i++] = stof(Mytext);
 }
 return arr;
}
