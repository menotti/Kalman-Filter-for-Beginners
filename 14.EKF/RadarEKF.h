

#include <iostream>
#include <cmath> /* sqrt() */ 

#include <stdlib.h> /* rand()*/

#include <iomanip> /*setw() */

#include <algorithm> 

#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"


#define DATA_TYPE float
using std::setw;


namespace blas = oneapi::mkl::blas;
namespace lapack = oneapi::mkl::lapack;
using namespace std;

auto nontransM = oneapi::mkl::transpose::nontrans;
auto    transM = oneapi::mkl::transpose::trans;

void inv(sycl::queue &queue, DATA_TYPE *A, int64_t N) {
    
    vector<sycl::event> event_list;

    try{

    std::int64_t scratch_size = lapack::getrf_scratchpad_size<DATA_TYPE>(queue, N, N, N);
    DATA_TYPE *scratchpad = sycl::malloc_shared<DATA_TYPE>(scratch_size+3, queue);
    std::int64_t *IPIV = sycl::malloc_shared<std::int64_t>(N, queue);
    
    auto getrf_ev = lapack::getrf(queue, N, N, A, N, IPIV, scratchpad, scratch_size);
    getrf_ev.wait();



//  cl::sycl::event getri(cl::sycl::queue &queue, std::int64_t n, T *a, std::int64_t lda, std::int64_t *ipiv, T *scratchpad, std::int64_t scratchpad_size, const std::vector<cl::sycl::event> &events = {})
    auto getri_ev = lapack::getri(queue, N, A, N, IPIV, scratchpad, 10);
    getri_ev.wait_and_throw();

    free(IPIV, queue);

    }catch(lapack::exception const &e){
        // Handle LAPACK related exceptions happened during synchronous call
        std::cout << "Unexpected exception caught during synchronous call to LAPACK API:\n"
                  << "info: " << e.info() << std::endl
                  << "detail:"<< e.detail() << std::endl; 
        // if (e.info() > 0) {
        // INFO is equal to the 'global' index of the element u_ii of the factor  
        // U which is equal to zero
            // info = e.info() + (n-2)*nb;
        std::terminate();
    }
} // end Inverse function



void display(DATA_TYPE *A, int l, int c){ 
	for(int i = 0; i< l;i++){
		for( int j =0;j<c;j++)
			cout << A[c*i+j] << " ";
		cout << endl;
	}	
    cout << endl;
}



void Hjacob(DATA_TYPE *H, DATA_TYPE *xhat){

    DATA_TYPE x1 = xhat[0];
    DATA_TYPE x3 = xhat[2]; 
    
    //dx(h)
    H[0] = x1 / sqrt(pow(x1, 2.0) + pow(x3, 2.0));
    H[1] = 0.0;
    H[2] = x3 / sqrt(pow(x1, 2.0) + pow(x3, 2.0));
    
    
} // end Hjacob

void Hx(DATA_TYPE *zp, DATA_TYPE *xhat){
    DATA_TYPE x1 = xhat[0];
    DATA_TYPE x3 = xhat[2];
    
    zp[0] = sqrt(pow(x1, 2.0) + pow(x3, 2.0));
}// end hx

class RadarEKF {
private: 
    sycl::queue queue;
    int M;
    int L;
    DATA_TYPE alpha = 1.0; 
    DATA_TYPE beta = 0.0;


    //Matrix Pointers
    DATA_TYPE *A;
    
    DATA_TYPE *Q;
    DATA_TYPE *R;
    DATA_TYPE *x;
    DATA_TYPE *P;
    DATA_TYPE *z;


    // Intermediary values to calculus
    DATA_TYPE *H;
    DATA_TYPE *xp;
    DATA_TYPE *Pp;
    DATA_TYPE *K;
    DATA_TYPE *AP;
    DATA_TYPE *PpHT;
    DATA_TYPE *HpHTR;
    DATA_TYPE *Hxp;
    DATA_TYPE *KH;

public: 
    RadarEKF(sycl::queue& queue, int M, int L);
    //modules
    void update(DATA_TYPE *);
    DATA_TYPE getResult( int );
    
    void setTransitionMatrix(DATA_TYPE *Aset);
    void setSttMeasure(DATA_TYPE * Hset);
    void setSttVariable(DATA_TYPE * xset);
    void setTransitionCovMatrix(DATA_TYPE * Qset);
    void setMeasureCovMatrix(DATA_TYPE * Rset);
    void setErrorCovMatrix(DATA_TYPE * P);

    //void ~RadarEKF()();
};
RadarEKF::RadarEKF(sycl::queue &queue,int M,int L): queue(queue), M(M), L(L)
{
    H     = sycl::malloc_shared<DATA_TYPE>(M * L, queue);
    xp    = sycl::malloc_shared<DATA_TYPE>(M * L, queue); 
    Pp    = sycl::malloc_shared<DATA_TYPE>(M * M, queue);
    K     = sycl::malloc_shared<DATA_TYPE>(M * L, queue);
    AP    = sycl::malloc_shared<DATA_TYPE>(M * M, queue);   
    PpHT  = sycl::malloc_shared<DATA_TYPE>(M * L, queue); 
    HpHTR = sycl::malloc_shared<DATA_TYPE>(L * L, queue);
    Hxp   = sycl::malloc_shared<DATA_TYPE>(L * L, queue);
    KH    = sycl::malloc_shared<DATA_TYPE>(M * M, queue); 
}

DATA_TYPE RadarEKF::getResult(int l){
    return x[l];
}

void RadarEKF::setTransitionMatrix(DATA_TYPE *Aset){
    A = Aset;
}

void RadarEKF::setSttMeasure(DATA_TYPE * Hset){
    H = Hset;
}

void RadarEKF::setSttVariable(DATA_TYPE * xset){
    x = xset;    
}

void RadarEKF::setTransitionCovMatrix(DATA_TYPE * Qset){
    Q = Qset;
}

void RadarEKF::setMeasureCovMatrix(DATA_TYPE * Rset){
    R = Rset;
}

void RadarEKF::setErrorCovMatrix(DATA_TYPE * Pset){
    P = Pset;
}      


void RadarEKF::update(DATA_TYPE *z_input){
    
    
    Hjacob(H, x);     
    z = z_input;
    
    
    constexpr int gemm_total = 10, axpy_total = 5;
    sycl::event gemm_task[gemm_total], scal_task;
    sycl::event axpy_task[axpy_total];
    vector<sycl::event> gemm[gemm_total];
       
     // xp(MxL) = A(MxM) * x(MxL) 
    blas::row_major::gemm(queue, nontransM, nontransM,
                          M, L, M, alpha, A, M, x, L, 
                          beta, xp, L).wait();

    // gemm_task[0].wait();
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
        // 2.1) PpHT(MxL) = Pp(MxM) * Ht(MxL) 
    gemm_task[3] = blas::row_major::gemm(queue, nontransM, transM, M, L, M, alpha, Pp, M, H, M, beta, PpHT, L, gemm[3]);
    gemm_task[3].wait();  
    
    
        // 2.2) HpHTR(LxL) = H(LxM) * [ Pp(MxM) * Ht(MxL) ] = H (LxM) * PpHT(MxL) 
    gemm_task[4] = blas::row_major::gemm(queue, nontransM, nontransM, L, L, M, alpha, H, M, PpHT, L, beta, HpHTR, L, gemm[4]);
    gemm_task[4].wait();

                                       
        // 2.3) HpHTR(LxL) = HpHTR(LxL) + R(LxL)
    axpy_task[1] = blas::axpy(queue, L*L, alpha, R, 1.0, HpHTR, 1.0);
    axpy_task[1].wait();

        // HpHTR(LxL) = inv(HpHTR)
    inv(queue, HpHTR, L);                                

         // 2.4) K(MxL) = (Pp(MxM) * Ht(MxM)) * HpHTR(MxM) -> PpHT(MxL) * HpHTR(LxL) 
    gemm_task[5] = blas::row_major::gemm(queue, nontransM, nontransM, M, L, L, alpha, PpHT, L, HpHTR, L, beta, K, L, gemm[5]);
    gemm_task[5].wait();

    
    
    // x(MxK) = xp(MxK) + K * (z - Hx(xp))
        // 3.1) Hxp(LxL) = Hx(xp(MxL))
    Hx(Hxp, xp); 
    
    
        // 3.2) z(LxL) = -Hxp(LxL) + z(LxL)
    axpy_task[2] = blas::axpy(queue, L*L, -alpha, Hxp, 1.0, z, 1.0);
    axpy_task[2].wait();    

    
        //3.3) // x(MxL) = K(MxL)*z(LxL)
    gemm_task[7] = blas::row_major::gemm(queue, nontransM, nontransM, M, L, L, alpha, K, L, z, L, beta, x, L, gemm[7]);
    gemm_task[7].wait();
 
    
    
    
        //3.4) x(MxL) = xp(MxL) + x(MxL)
    axpy_task[3] = blas::axpy(queue, M*L, alpha, xp, 1.0, x, 1.0);
    axpy_task[3].wait();
    
    
    
    // P = Pp - K * H * Pp
        //4.1) KH(MxM) = K(MxL) * H(LxM)
    gemm_task[8] = blas::row_major::gemm(queue, nontransM, nontransM, M, M, L, alpha, K, L, H, M, beta, KH, M, gemm[8]);
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
RadarEKF::~RadarEKF(){
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

