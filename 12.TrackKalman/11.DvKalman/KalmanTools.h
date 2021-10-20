

#include <iostream>

#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <limits>
#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"

#include <algorithm> 

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
    int M = 2;
    int N = 1;
    double dt;
    const int size_1 = N*N;
    const int size_2 = M*N;
    const int size_3 = M*M;

    
    //Matrix Declaration
    double *A = sycl::malloc_shared<double>(size_3, queue);
    double *H = sycl::malloc_shared<double>(size_2, queue); 
    double *Q = sycl::malloc_shared<double>(size_3, queue); 
    double *R = sycl::malloc_shared<double>(size_1, queue);  
    double *x = sycl::malloc_shared<double>(size_2, queue); 
    double *P = sycl::malloc_shared<double>(size_3, queue); 
    double *z = sycl::malloc_shared<double>(size_1, queue); 

    // Intermediary values to calculus
    double *xp = sycl::malloc_shared<double>(size_2, queue);
    double *Pp = sycl::malloc_shared<double>(size_3, queue);
    double *K = sycl::malloc_shared<double>(size_2, queue);
    double *AP = sycl::malloc_shared<double>(size_3, queue);   
    double *PpHT = sycl::malloc_shared<double>(size_2, queue); 
    double *HpHTR = sycl::malloc_shared<double>(size_1, queue);
    double *Hxp = sycl::malloc_shared<double>(size_3, queue);
    double *Kz = sycl::malloc_shared<double>(size_3, queue);
    double *KH = sycl::malloc_shared<double>(size_3, queue); 
    
public: 
    //modules
    void default_values(double delta_t);
    void filter(double z_value);
    double getResult(int row, int col);

    void end_task();

};



void Kalman_filter::default_values(double delta_t){
//define initial values to the Kalman Implementation in accord to Phil Kim's book
    
	dt = delta_t;
	A[0] = 1.0; A[1] = dt;
    A[2] = 0.0; A[3] = 1.0;
    
    x[0] = 0.0; x[1] = 20.0;
    
    H[0] = 1.0; H[1] = 0.0;
    
    Q[0] = 1.0; Q[1] = 0.0;
    Q[2] = 0.0; Q[3] = 3.0;
    
    R[0] = 10.0;
    
    P[0] = 5.0; P[1] = 0.0;
    P[2] = 0.0; P[3] = 5.0;


    std::cout<<"Default Values inserted succefully"<<endl;
}

void Kalman_filter::filter(double z_value){
    z[0] = z_value;
    constexpr int gemm_total = 10, axpy_total = 5;
    sycl::event gemm_task[gemm_total], scal_task;
    sycl::event axpy_task[axpy_total];
    std::cout<<std::showpoint;
    vector<sycl::event> gemm[gemm_total];
       
     // xp(MxN) = A(MxM) * x(MxN)     
    gemm_task[0] = blas::row_major::gemm(queue, nontransM, nontransM, M, N, M, alpha, A, M, x, N, beta, xp, N, gemm[0]);
        cout << "matriz X:" << endl;
    cout << x[0] << " " << x[1] << endl;
    gemm_task[0].wait();
    cout << "matriz A:" << endl;
    cout << A[0] << " " << A[1] << endl;
    cout << A[2] << " " << A[3] << endl;
    
     // Pp(MxM) = A * P * A' + Q(MxM)
        //1.1) AP(MxM) = A(MxM) * P(MxM)
    gemm_task[1] = blas::row_major::gemm(queue, nontransM, nontransM, M, M, M, alpha, A, M, P, M, beta, AP, M, gemm[1]);
    gemm_task[1].wait();
    
        //1.2) Pp = AP(MxM) * A'(MxM)
    gemm_task[2] = blas::row_major::gemm(queue, nontransM, transM, M, M, M, alpha, AP, M, A, M, beta, Pp, M, gemm[2]);
    gemm_task[2].wait();

       //1.3) Pp(MxM) = Pp(MxM) + Q(MxM) 
    axpy_task[0] = blas::axpy(queue, size_3, alpha, Q, 1.0, Pp, 1.0);
    axpy_task[0].wait();

    // K = Pp * H' * inv(H * Pp * H' + R)
        // 2.1) PpHT(MxN) = Pp(MxM) * H'(MxN)
    gemm_task[3] = blas::row_major::gemm(queue, nontransM, transM, M, N, M, alpha, Pp, M, H, M, beta, PpHT, N, gemm[3]);
    gemm_task[3].wait();

    
        // 2.2) HpHTR(NxN) = H(NxM) * [ Pp(MxM) * Ht(MxN) ] = H (NxM) * PpHT(MxN) 
    gemm_task[4] = blas::row_major::gemm(queue, nontransM, nontransM, N, N, M, alpha, H, M, PpHT, N, beta, HpHTR, N, gemm[4]);
    gemm_task[4].wait();
                                       
        // 2.3) HpHTR(NxN) = HpHTR(NxN) + R(NxN)
    axpy_task[1] = blas::axpy(queue, size_1, alpha, R, 1.0, HpHTR, 1.0);
    axpy_task[1].wait();
    
        // HpHTR(NxN) = inv(HpHTR)
    inv(queue, HpHTR, N);                                 
    
         // 2.4) K(MxN) = (Pp(MxM) * Ht(MxN)) * HpHTR(NxN) -> PpHT(MxN) * HpHTR(NxN)
    gemm_task[5] = blas::row_major::gemm(queue, nontransM, nontransM, M, N, N, alpha, PpHT, N, HpHTR, N, beta, K, N, gemm[5]);
    gemm_task[5].wait();   
        
    // x(MxN) = xp(MxN) + K * (z - H * xp)          
        // 3.1) Hxp(NxN) = H(NxM) * xp(MxN)
    gemm_task[6] = blas::row_major::gemm(queue, nontransM, nontransM, N, N, M, alpha, H, M, xp, N, beta, Hxp, N, gemm[6]);
    gemm_task[6].wait();
    
        // 3.2) z = -Hxp(NxN) + z(NxN)
    axpy_task[2] = blas::axpy(queue, size_1, -alpha, Hxp, 1.0, z, 1.0);
    axpy_task[2].wait();
    
                                                                      
        //3.3) // x(MxN) = K(MxN)*z(NxN)
    gemm_task[7] = blas::row_major::gemm(queue, nontransM, nontransM, M, N, N, alpha, K, N, z, N, beta, x, N, gemm[7]);
    gemm_task[7].wait();
    
        //3.4) x(MxN) = xp(MxN) + x(MxN) 
    axpy_task[3] = blas::axpy(queue, size_2, alpha, xp, 1.0, x, 1.0);
    axpy_task[3].wait();
    
    // P = Pp - K * H * Pp
        //4.1) KH(MxM) = K(MxN)*H(NxM)
    gemm_task[8] = blas::row_major::gemm(queue, nontransM, nontransM, M, M, N, alpha, K, N, H, M, beta, KH, M, gemm[8]);
    gemm_task[8].wait();

        //4.2) P =(-1)* KH(MxM) * Pp(MxM)
    gemm_task[9] = blas::row_major::gemm(queue, nontransM, nontransM, M, M, M, -alpha, KH, M, Pp, M, beta, P, M, gemm[9]);
    gemm_task[9].wait();
    
        //4.3) P = (Pp - P)
    axpy_task[4] = blas::axpy(queue, size_3, alpha, Pp, 1.0, P, 1.0);
    axpy_task[4].wait();
    
    //End calculus here, then its necessary to acess it by GetResult, 
    //which is obtained by the matrix X(nRow x nCol).
    
    cout << "matriz xp:" << endl;
    cout << xp[0] << " " << xp[1] << endl;
    cout << endl;
    
    cout << "matriz K:" << endl;
    cout << K[0] << " " << K[1] << endl;
    
    cout << endl;

}    

    
double Kalman_filter::getResult(int row, int col){
    bool null_number = row<0 || col<0; 
    bool overflow = row>M || col>N;
        if (!overflow && !null_number) return x[col + row*M];        
        else {
            cout<<"Error: Memory not located"<<endl; 
            return -1;
        }
}




/*void end_task(){
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

