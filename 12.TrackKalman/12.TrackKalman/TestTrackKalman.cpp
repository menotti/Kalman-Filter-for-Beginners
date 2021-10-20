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

#include <limits>
#include <random>
#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"
namespace blas = oneapi::mkl::blas;
namespace lapack = oneapi::mkl::lapack;

#include "KalmanTools.h"


// Struct to store measures (x,y):
struct coordenadas
{
	long double X ;
	long double Y;

};

//Function Prototype
coordenadas *GetCoordenadas();


int main(){
    //create file : TestTrackFile.csv
    ofstream myfile("TestTrack.csv");
    myfile << "Xmsaved,Ymsaved,Xhsaved,Yhsaved\n";
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
        double dt = 1;
        const int Nsamples = 24;
        
        double t[Nsamples];
        Kalman_filter *kalman = new Kalman_filter();
        auto *Xmsaved = sycl::malloc_shared<double>(Nsamples, queue);
        auto *Ymsaved = sycl::malloc_shared<double>(Nsamples, queue);
        
        auto *Xhsaved = sycl::malloc_shared<double>(Nsamples, queue);
        auto *Yhsaved = sycl::malloc_shared<double>(Nsamples, queue);
        
        
        auto *A = sycl::malloc_shared<double>(4 * 4, queue);
        A[0]  = 1; A[1] = dt; A[2]  = 0; A[3]  = 0;
        A[4]  = 0; A[5]  = 1; A[6]  = 0; A[7]  = 0;
        A[8]  = 0; A[9]  = 0; A[10] = 1; A[11] =dt;
        A[12] = 0; A[13] = 0; A[14] = 0; A[15] = 1;
        
        
        auto *H = sycl::malloc_shared<double>(2 * 4, queue);
        H[0]  = 1; H[1] = 0; H[2] = 0; H[3] = 0;
        H[4]  = 0; H[5] = 0; H[6] = 1; H[7] = 0;
        
        auto *Q = sycl::malloc_shared<double>(4 * 4, queue);
        Q[0]  = 1; Q[1]  = 0; Q[2]  = 0; Q[3]  = 0;
        Q[4]  = 0; Q[5]  = 1; Q[6]  = 0; Q[7]  = 0;
        Q[8]  = 0; Q[9]  = 0; Q[10] = 1; Q[11] = 0;
        Q[12] = 0; Q[13] = 0; Q[14] = 0; Q[15] = 1;
        
        auto *R = sycl::malloc_shared<double>(2 * 2, queue);
        R[0] = 50; R[1] = 0;
        R[2] = 0; R[3] = 50;     
        
        auto *x = sycl::malloc_shared<double>(4 * 1, queue);
        x[0] = 0; x[1] = 0; x[2]  = 0; x[3] = 0;     
        
        auto *P = sycl::malloc_shared<double>(4 * 4, queue);
        P[0] =100; P[1]  = 0; P[2]  = 0; P[3]  = 0;
        P[4]  = 0; P[5] =100; P[6]  = 0; P[7]  = 0;
        P[8]  = 0; P[9]  = 0; P[10]=100; P[11] = 0;
        P[12] = 0; P[13] = 0; P[14] = 0; P[15]=100;
        kalman->setDeltaT(dt);
        kalman->setTransitionMatrix(A);
        kalman->setSttMeasure(H);
        kalman->setSttVariable(x);
        kalman->setTransitionCovMatrix(Q);
        kalman->setMeasureCovMatrix(R);
        kalman->setErrorCovMatrix(P);
        //generate list with Coordinate measures
        coordenadas *Measure = GetCoordenadas();
        
        
        if (!Yhsaved || !Xhsaved) {
            cerr << "Could not allocate memory for vectors." << endl;
            exit(1);
        }
        
        int first = 0;
        double xm, ym, xh, yh;
        for (int i = 0; i < Nsamples; i++){  
            //Store initial values obtained previously
            t[i] = i;
            xm = Xmsaved[i] = Measure[i].X;
            ym = Ymsaved[i] = Measure[i].Y;
            
            //calculate, then store on arrays
            kalman->filter(xm, ym);
            xh = Xhsaved[i] = kalman->getResult(0,0);
            yh = Yhsaved[i] = kalman->getResult(2,0);
            
            //Save the results below in myfile. 
            myfile << xm <<", "<< ym << ", " << xh << ", " << yh <<'\n';
        }
        myfile.close();
        free(Yhsaved, queue);
        free(Xhsaved, queue);
        kalman->end_task();

    } catch (const exception &e) {
        cerr << "An exception occurred: "
                  << e.what() << endl;
        exit(1);
    }
}

coordenadas *GetCoordenadas(){
	string txt, name = "Measure_Img.csv";
	const int Ncoord = 24;
	coordenadas *Coord = new coordenadas[Ncoord];
	coordenadas *Pointer = Coord; 
	std::ifstream Arquivo(name);
	long double x , y;
	int pos, len , i = 0;
	while(getline(Arquivo, txt)){
		
		pos = txt.find(",");
		len = txt.length();
		x = std::stold(txt.substr(0, pos));

		y = std::stold(txt.substr(pos+1, len-1));
		Coord[i].X = x;
		Coord[i].Y = y;
		i++;
		}
		return Pointer;
}
