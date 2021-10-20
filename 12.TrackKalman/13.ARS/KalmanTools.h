

#include <iostream>
#include <cmath>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <limits>
#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"

#include <algorithm> 

#include <iomanip>
using std::setw;

namespace blas = oneapi::mkl::blas;
namespace lapack = oneapi::mkl::lapack;
using namespace std;

auto nontransM = oneapi::mkl::transpose::nontrans;
auto    transM = oneapi::mkl::transpose::trans;

void inv(sycl::queue &queue, double *A, int64_t N) {
    sycl::event getr_task[2];
    vector<sycl::event> event_list;

    double scratch_size = lapack::getrf_scratchpad_size<double>(queue, N, N, N);
    double *scratchpad = sycl::malloc_shared<double>(scratch_size+1, queue);

    auto *IPIV = sycl::malloc_shared<int64_t>(N*N, queue);
    
    getr_task[0] = lapack::getrf(queue, N, N, A, N, IPIV, scratchpad, scratch_size, event_list);
    getr_task[0].wait();
    getr_task[1] = lapack::getri(queue, N, A, N, IPIV, scratchpad, scratch_size, event_list);
    getr_task[1].wait();
    
    free(IPIV, queue);
}
class Kalman_filter {
private: 
    sycl::queue queue;
    double alpha = 1.0; 
    double beta = 0.0;
    int M = 4;
    int N = 2;
    int L = 1; 
    double dt;
    
    const int size_1 = N*N;
    const int size_2 = M*N;
    const int size_3 = M*M;
    
    const int size_4 = M*L;

    //Matrix Pointers
    double *A;
    double *H;
    double *Q;
    double *R;
    double *x;
    double *P;
    double *z;


    // Intermediary values to calculus
    double *xp    = sycl::malloc_shared<double>(M * L, queue);
    double *Pp    = sycl::malloc_shared<double>(M * M, queue);
    double *K     = sycl::malloc_shared<double>(M * M, queue);
    double *AP    = sycl::malloc_shared<double>(M * M, queue);   
    double *PpHT  = sycl::malloc_shared<double>(M * M, queue); 
    double *HpHTR = sycl::malloc_shared<double>(M * M, queue);
    double *Hxp   = sycl::malloc_shared<double>(M * L, queue);
    double *KH    = sycl::malloc_shared<double>(M * M, queue); 

public: 
    //modules
    void filter(double *,double *);
    double getResult(int row, int col);
    
   void setDeltaT( double setdt );
    void setTransitionMatrix(double *Aset);
    void setSttMeasure(double * Hset);
    void setSttVariable(double * xset);
    void setTransitionCovMatrix(double * Qset);
    void setMeasureCovMatrix(double * Rset);
    void setErrorCovMatrix(double * P);
    double get_phi();
    double get_theta();
    double get_psi();
    void end_task();
};



void Kalman_filter::setDeltaT( double setdt ){
    dt = setdt;
}

void Kalman_filter::setTransitionMatrix(double *Aset){
    A = Aset;
}

void Kalman_filter::setSttMeasure(double * Hset){
    H = Hset;
}

void Kalman_filter::setSttVariable(double * xset){
    x = xset;    
}

void Kalman_filter::setTransitionCovMatrix(double * Qset){
    Q = Qset;
}

void Kalman_filter::setMeasureCovMatrix(double * Rset){
    R = Rset;
}

void Kalman_filter::setErrorCovMatrix(double * Pset){
    P = Pset;
}      

double Kalman_filter::get_phi(){
    double x1,x2,x3,x4;
    x1 = x[0]; x2 = x[1];
    x3 = x[2]; x4 = x[3];
    
    double phi = atan2(2*(x3*x4 + x1*x2), 1 - 2*(x2*x2+x3*x3));
    return phi;
}      
double Kalman_filter::get_theta(){
    double x1,x2,x3,x4;
    x1 = x[0]; x2 = x[1];
    x3 = x[2]; x4 = x[3];
    
    double theta = -asin( 2*(x2*x4 - x1*x3));
    return theta;
}      
double Kalman_filter::get_psi(){
    double x1,x2,x3,x4;
    x1 = x[0]; x2 = x[1];
    x3 = x[2]; x4 = x[3];
    
    double psi   =  atan2(2*(x2*x3 + x1*x4), 1 - 2*(x3*x3+x4*x4));
    return psi;
}      


void Kalman_filter::filter(double *A_input,double *z_input){
    z = z_input;
    setTransitionMatrix(A_input); 
    
    constexpr int gemm_total = 10, axpy_total = 5;
    sycl::event gemm_task[gemm_total], scal_task;
    sycl::event axpy_task[axpy_total];
    vector<sycl::event> gemm[gemm_total];
       
     // xp(MxL) = A(MxM) * x(MxL) 
    gemm_task[0] = blas::row_major::gemm(queue, nontransM, nontransM, M, L, M, alpha, A, M, x, L, beta, xp, L, gemm[0]);
    gemm_task[0].wait();
       
    
     // Pp(MxM) = A * P * A' + Q(MxM) 
        //1.1) AP(MxM) = A(MxM) * P(MxM)
    gemm_task[1] = blas::row_major::gemm(queue, nontransM, nontransM, M, M, M, alpha, A, M, P, M, beta, AP, M, gemm[1]);
    gemm_task[1].wait();
       
        //1.2) Pp = AP(MxM) * A'(MxM) 
    gemm_task[2] = blas::row_major::gemm(queue, nontransM, transM, M, M, M, alpha, AP, M, A, M, beta, Pp, M, gemm[2]);
    gemm_task[2].wait();
    
    
        //1.3) Pp(MxM) = Pp(MxM) + Q(MxM)  
    axpy_task[0] = blas::axpy(queue, M*M, alpha, Q, 1.0, Pp, 1.0);
    axpy_task[0].wait();
    
    
    // K = Pp * H' * inv(H * Pp * H' + R)
        // 2.1) PpHT(MxM) = Pp(MxM) * Ht(MxM) 
    gemm_task[3] = blas::row_major::gemm(queue, nontransM, transM, M, M, M, alpha, Pp, M, H, M, beta, PpHT, M, gemm[3]);
    gemm_task[3].wait();

        // 2.2) HpHTR(NxN) = H(MxM) * [ Pp(MxM) * Ht(MxM) ] = H (MxM) * PpHT(MxM) 
    gemm_task[4] = blas::row_major::gemm(queue, nontransM, nontransM, M, M, M, alpha, H, M, PpHT, M, beta, HpHTR, M, gemm[4]);
    gemm_task[4].wait();

                                       
        // 2.3) HpHTR(MxM) = HpHTR(MxM) + R(MxM)
    axpy_task[1] = blas::axpy(queue, M*M, alpha, R, 1.0, HpHTR, 1.0);
    axpy_task[1].wait();

    
        // HpHTR(MxM) = inv(HpHTR)
    inv(queue, HpHTR, M);                                
    
         // 2.4) K(MxM) = (Pp(MxM) * Ht(MxM)) * HpHTR(MxM) -> PpHT(MxM) * HpHTR(MxM) 
    gemm_task[5] = blas::row_major::gemm(queue, nontransM, nontransM, M, M, M, alpha, PpHT, M, HpHTR, M, beta, K, M, gemm[5]);
    gemm_task[5].wait();
    
    
    
    
        // x(MxK) = xp(MxK) + K * (z - H * xp)
        // 3.1) Hxp(MxL) = H(MxM) * xp(MxL)
    gemm_task[6] = blas::row_major::gemm(queue, nontransM, nontransM, M, L, M, alpha, H, M, xp, L, beta, Hxp, L, gemm[6]);
    gemm_task[6].wait();
    

    
        // 3.2) z(MxL) = -Hxp(MxL) + z(MxL)
    axpy_task[2] = blas::axpy(queue, M*L, -alpha, Hxp, 1.0, z, 1.0);
    axpy_task[2].wait();    
    
        //3.3) // x(MxL) = K(MxM)*z(MxL)
    gemm_task[7] = blas::row_major::gemm(queue, nontransM, nontransM, M, L, M, alpha, K, M, z, L, beta, x, L, gemm[7]);
    gemm_task[7].wait();
    
        //3.4) x(MxL) = xp(MxL) + x(MxL)
    axpy_task[3] = blas::axpy(queue, M*L, alpha, xp, 1.0, x, 1.0);
    axpy_task[3].wait();
    
    // P = Pp - K * H * Pp
        //4.1) KH(MxM) = K(MxM)*H(MxM)
    gemm_task[8] = blas::row_major::gemm(queue, nontransM, nontransM, M, M, M, alpha, K, M, H, M, beta, KH, M, gemm[8]);
    gemm_task[8].wait();

        //4.2) P(MxM) =(-1)* KH(MxM) * Pp(MxM) 
    gemm_task[9] = blas::row_major::gemm(queue, nontransM, nontransM, M, M, M, -alpha, KH, M, Pp, M, beta, P, M, gemm[9]);
    gemm_task[9].wait();
    
        //4.3) P(MxM) = (Pp - P) 
    axpy_task[4] = blas::axpy(queue, M * M, alpha, Pp, 1.0, P, 1.0);
    axpy_task[4].wait();
    


    //End calculus here, then its necessary to acess it by GetResult, 
    //which is obtained by the matrix X(nRow x nCol).
    
    
}    



/*
void end_task(){
    free(A, queue);
    free(H, queue);
    free(Q, queue);
    free(R, queue);
    free(x, queue);
    free(P, queue);
    free(z, queue);

    free(xp, queue);
    free(Pp, queue);
    free(K, queue);
    free(AP, queue);
    free(PpHT, queue);
    free(HpHTR, queue);
    free(Hxp, queue);
    free(Kz, queue);
    free(KH, queue);
    
}
*/

