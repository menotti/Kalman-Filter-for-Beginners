#ifndef LIBKF_HPP
#define LIBKF_HPP

//Built-in Libs
#include <iostream> 
#include <vector>
#include <cstring>
#include <cmath>

//Local Libs
#include "Matrix.hpp"


//Define value_type 
#ifndef DOUBLE_PRECISION
typedef float _Tp;
#else 
typedef double T;
#endif



struct wm {
        Matrix<_Tp> *Q;
        Matrix<_Tp> *Qeta;
        Matrix<_Tp> *Qeta_sr;
        Matrix<_Tp> *R;
        Matrix<_Tp> *R_sr;
        Matrix<_Tp> *P;
        Matrix<_Tp> *P_sr;
        Matrix<_Tp> *Pp;

        Matrix<_Tp> *Pz;
        Matrix<_Tp> *Pxz;
        std::vector<_Tp> *W;
        _Tp kappa;
};

// template<class T>
struct ss_d{
        Matrix<_Tp> *x;
        Matrix<_Tp> *xp;
        Matrix<_Tp> *z;
        Matrix<_Tp> *zp; 

        Matrix<_Tp> *Phi;// Qual seria essa matrix? 
        Matrix<_Tp> *G;  // Same here  
        Matrix<_Tp> *H;  // idem here 

        //Additional Matrices
        Matrix<_Tp> *Xi;
        Matrix<_Tp> *fXi;
        Matrix<_Tp> *hXi;
        Matrix<_Tp> *K;
        _Tp dt;
        int x_states;
        int z_states;

        
};

//Functions pre-declarations
void fx(Matrix<_Tp> *Xi, double dt, Matrix<_Tp> *fXi, int itr, int x_stt);

void hx(Matrix<_Tp> *Xi, Matrix<_Tp> *hXi, int itr);

void set_weights(std::vector<_Tp> *W, _Tp kappa, int stt);

void sigma_pts(ss_d *ss, wm *wm);

// template<_Tpypename Tp>
void UT(Matrix<_Tp> *Xi, std::vector<_Tp> *W, Matrix<_Tp> *NoiseCov,
        Matrix<_Tp> *xm, Matrix<_Tp> *xcov, int n);


ss_d *init_ssd(const int x_stt, const int z_stt,_Tp dt);

wm *init_wm(const int x_stt, const int z_stt,_Tp kappa);

void ukf_predict(struct wm *wm, struct ss_d *ss);

void ukf_update(struct wm *wm, struct ss_d *ss);

#endif //LIBKF_HPP