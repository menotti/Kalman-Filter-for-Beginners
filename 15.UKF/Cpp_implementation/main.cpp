
// Built-in Libs
#include <iostream>
#include <vector>
#include <random>  /* using default_random_engine */
#include <fstream> /* using ofstream*/

//Local Libs
#include "Matrix.hpp"

#include "libkf.hpp"

typedef float value_type;

typedef std::vector<double> DVector;
// typedef wm<double> Dwm;

//Function Prototype
double *GetRadar(int Nsamples,double dt, int seed);


//Set some const values


//Total x states: 3
const int L = 3;
//Total y states: 1
const int M = 1;

const value_type kappa = 0.0;
const value_type dt = 0.05;
//Total Samples: 400
const int Nsamples = 2000/dt;

const int seed = 777;

int main()
{

    //Declare Weight Matrices & states
    struct ss_d *ss = init_ssd(L, M, dt);
    struct wm *wm = init_wm(L, M, kappa);

    wm->Q->eye(0.01);
    wm->R->at(0,0) = 100.0;
    wm->P->eye(100.0);

    ss->x->at(0,0) = 0.0;
    ss->x->at(0,1) = 90.0;
    ss->x->at(0,2) = 1100.0;

    // Creating file -- TestRadar.csv
    std::ofstream myfile("TestRadarUKF.csv");
    myfile << "time,Position,Velocity,Altitude\n";

    //Build up vector -- simulate radar data
    double *R = GetRadar(Nsamples, dt, seed);
    double t, pos, vel, alt;


    for(int i = 0; i < Nsamples; i++)
    {
        //transfer read val to Z matrix
        ss->z->at(0,0) = R[i];
        t = dt*i;
        ukf_predict(wm, ss);
        ukf_update(wm, ss);

        pos = ss->x->at(0,0);
        vel = ss->x->at(1,0);
        alt = ss->x->at(2,0);

        //#Save the results below in myfile. 
        myfile << t <<", "<< pos << ", " << vel << ", " << alt <<'\n';        
    }

    return  0;
}

double *GetRadar(int Nsamples,double dt, int seed){
	int i = 0; 

	std::default_random_engine generator;
	generator.seed(seed);
	std::normal_distribution<double> distribution(0.0, 1.0);

	double *radar = new double[Nsamples];

	double vel, v, r, alt, pos, posp=0;
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