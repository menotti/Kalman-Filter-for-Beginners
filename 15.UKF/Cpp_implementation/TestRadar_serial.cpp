#ifndef LIBKF_HPP
#define LIBKF_HPP

// Built-in Libs
#include <iostream>
#include <vector>


//Local Libs
#include "include/Matrix.hpp"
#include "include/Matrix.cpp"

typedef Matrix<double> DMatrix;
typedef std::vector<double> DVector;

//Total x states: 3
static const int n = 3;
//Total y states: 1
static const int m = 1;

int main()
{
    //Declare Initial Matrices
    DMatrix x(n, 1);
    DMatrix tst(3,8);
    DMatrix Q(n, n);
    DMatrix R(1, 1); 
    DMatrix P(1, 1);

 
    x.at(0, 0) =    0.0;
    x.at(1, 0) =   90.0; 
    x.at(2, 0) = 1100.0; 

    Q.eye(0.01);
    R.at(0,0) = 100.0;
    P.eye(100.0);


    //Set some const values
    double kappa = 0.0;
    double dt = 0.0;
    
    tst.display();

    return  0;
}

#endif //LIBKF_HPP