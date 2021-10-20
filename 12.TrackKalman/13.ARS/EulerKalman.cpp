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
struct Gyro
{
	long double coord_x;
	long double coord_y;
	long double coord_z;
};


//Function Prototype
Gyro *GetRead(string filename, int Samples);
void eye(double* A, int l, double a );
void zero(double* A, int l, int m);
void AdjustMatrix(double *,double, double,double,double);
void EulerToQuaternion(double *, double, double, double);
void EulerAccel(double, double, double &ax, double &ay);

int main(){
    //create file : EulerKalman.csv
    ofstream myfile("EulerSaved.csv");
    myfile << "time,Phisaved,Thetasaved,Psisaved\n";
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
        
         
        //About Accelerator Device & Queue
        sycl::device device = sycl::device(sycl::default_selector());
        cout << "Device: " << device.get_info<sycl::info::device::name>() << "\n";
        sycl::queue queue(device, async_handler);
        vector<sycl::event> event_list;
        
        
        
        //Declare example variables, lists and Kalman Object
        double dt = 0.01;
        const int Nsamples = 41500;
        double t[Nsamples];

        auto *Phisaved = sycl::malloc_shared<double>(Nsamples, queue);
        auto *Psisaved = sycl::malloc_shared<double>(Nsamples, queue);
        auto *Thetasaved = sycl::malloc_shared<double>(Nsamples, queue);
        
        
        
        auto *A = sycl::malloc_shared<double>(4 * 4, queue);
        zero(A,4,4);
        auto *H = sycl::malloc_shared<double>(4 * 4, queue);
        eye(H, 4, 1.0);
        
        auto *Q = sycl::malloc_shared<double>(4 * 4, queue);
        eye(Q, 4, 0.0001);
        
        auto *R = sycl::malloc_shared<double>(4 * 4, queue);
        eye(R, 4, 10.0);
        
        auto *x = sycl::malloc_shared<double>(4 * 1, queue);
        x[0] = 1; x[1] = 0; x[2]  = 0; x[3] = 0;     
        
        auto *P = sycl::malloc_shared<double>(4 * 4, queue);
        eye(P, 4, 1.0);
        
        auto *z = sycl::malloc_shared<double>(4 * 1, queue);
        
        Kalman_filter *kalman = new Kalman_filter();
        kalman->setTransitionCovMatrix(Q);
        kalman->setMeasureCovMatrix(R);
        kalman->setSttMeasure(H);
        kalman->setErrorCovMatrix(P);
        kalman->setSttVariable(x);
        kalman->setDeltaT(dt);
        
        
        //generate list with Coordinate measures
        Gyro *Radius = GetRead("ArsGyro.csv", Nsamples);
        Gyro *angular_accel = GetRead("ArsAccel.csv", Nsamples);
        
        
        if (!Phisaved || !Psisaved || !Thetasaved) {
            cerr << "Could not allocate memory for vectors." << endl;
            exit(1);
        }
        
        int first = 0;
        double p, q, r , ax, ay, phi, theta, psi;
        for (int i = 0; i < Nsamples; i++){  
            //Get values from Gyro Struct and define t array
            t[i] = i*dt;
            
            p = Radius[i].coord_x;
            q = Radius[i].coord_y;
            r = Radius[i].coord_z;
            
            ax = angular_accel[i].coord_x;
            ay = angular_accel[i].coord_y;
            
            AdjustMatrix(A, p, q, r, dt);
            //These values are returned in ax and ay.
            EulerAccel(ax,ay, phi, theta); 
            EulerToQuaternion(z, phi, theta, 0);
            
            kalman->filter(A,z);
            phi = kalman->get_phi();
            theta = kalman->get_theta();
            psi = kalman->get_psi();
            
            //calculate, then store on arrays
            phi   = phi*180/3.141592;
            theta = theta*180/3.141592;
            psi   = psi*180/3.141592;
            Phisaved[i]   = psi; 
            Thetasaved[i] = theta;
            Psisaved[i]   = psi;
            //Save the results below in myfile. 
            myfile << t[i] <<", "<< phi << ", " << theta << ", " << psi <<'\n';
        }
        myfile.close();
        free(Phisaved, queue);
        free(Psisaved, queue);
        free(Thetasaved, queue);
        //kalman->end_task();

    } catch (const exception &e) {
        cerr << "An exception occurred: "
                  << e.what() << endl;
        exit(1);
    }
}



Gyro *GetRead(string filename, int Samples){
	
	ifstream file;
	file.open(filename);
	Gyro *L = new Gyro[Samples];
	if(file.is_open()){
		string buff, out_x, out_y,out_z;
		size_t pos_1, pos_2;
		int it = 0;
		cout << "reading doc.: " << filename;

		while(getline(file, buff)){;
			pos_1 = buff.find(",");
			pos_2 = buff.rfind(",");
			// Essa condicao serve para leitura de ambos os arquivos
			if(pos_1 != pos_2){
				L[it].coord_x = stold(buff.substr(0,pos_1));
				L[it].coord_y = stold(buff.substr(1+pos_1, pos_2-pos_1));
				L[it].coord_z = stold(buff.substr(1+pos_2));
			} else { 
				//Condicao para leitura do Arq. ArsAccel.csv
				L[it].coord_x = stold(buff.substr(0,pos_1));
				L[it].coord_y = stold(buff.substr(1+pos_1));

			}
			it++;
		}
		cout << ": OK! " << endl;
	} else {
		cout << "Error : Could not read !!" <<endl;
        L = nullptr;
	}
	return L;

}

void eye(double* A, int l, double val){
	for(int i =0; i < l; i++){
		for(int j=0; j<l; j++){
			if(i == j) A[j+i*l] = val;
			else A[j+i*l] = 0.0;
		}
	}
}
void zero(double* A, int l, int m){
	for(int i =0; i < l; i++){
		for(int j=0; j<m; j++){
			A[j+i*l] = 0.0;
		}
	}
}


void EulerToQuaternion(double *z, double u, double v, double w){
	
	double Phi, Theta, Psi;

	Phi   = u / 2;
    Theta = v / 2;
    Psi   = w / 2;
    z[0] = cos(Phi)*cos(Theta)*cos(Psi)+sin(Phi)*sin(Theta)*sin(Psi);
    z[1] = sin(Phi)*cos(Theta)*cos(Psi)-cos(Phi)*cos(Theta)*sin(Psi);
    z[2] = cos(Phi)*sin(Theta)*cos(Psi)+sin(Phi)*sin(Theta)*sin(Psi);
    z[3] = cos(Phi)*cos(Theta)*sin(Psi)-sin(Phi)*sin(Theta)*cos(Psi);	    
}


void EulerAccel(double ax, double ay, double &p,double &t){
	double g = 9.8;
    
	t = asin(ax / g);
	p = asin(-ay / (g * cos(ax)));


}


void AdjustMatrix(double *aux,double p, double q, double r, double dt){
	aux[0] = 0; aux[1] =-p; aux[2] =-q; aux[3] =-r;
	aux[4] = p; aux[5] = 0; aux[6] = r; aux[7] =-q;
	aux[8] = q; aux[9] =-r; aux[10]= 0; aux[11]= p;
	aux[12]= r; aux[13]= q; aux[14]=-p; aux[15]= 0;
	for(int i =0; i <4; i++) {
		for(int j=0; j < 4; j++){
			aux[j+4*i] = aux[j+4*i]*0.5*dt;
			if(i==j) aux[j+4*i]+= 1;
		}
	}
}
