

#include <iostream> 


#include <cmath>

#include <stdlib.h>

#include <limits>


#include <CL/sycl.hpp>
using namespace sycl;

#include "oneapi/mkl.hpp"

#include <algorithm> 


namespace blas = oneapi::mkl::blas;
namespace lapack = oneapi::mkl::lapack;


auto nontransM = oneapi::mkl::transpose::nontrans;
auto    transM = oneapi::mkl::transpose::trans;
auto    upperM = oneapi::mkl::uplo::upper;
auto    lowerM = oneapi::mkl::uplo::lower;
                    
 

void display(double *A, int l, int c){ 
	for(int i = 0; i< l;i++){
		for( int j =0;j<c;j++)
			std::cout << A[c*i+j] << " ";
		std::cout << std::endl;
	}	

}

void zero(double* A, int l, int m){
	for(int i =0; i < l*m; i++){
			A[i] = 0.0;
	}
}

void eye(double* A, int l, double val){
	for(int i =0; i < l; i++){
		for(int j=0; j<l; j++){
			if(i == j) A[j+i*l] = val;
			else A[j+i*l] = 0.0;
		}
	}
}

double* matrix2vector(sycl::queue &queue, double* A, int m, int n, int pos){
    
    double *y = sycl::malloc_shared<double>(n, queue);
    for(int i = 0; i < n; i++)
        y[i] = A[n*pos + i];

    return y;
}

double* NullVector(sycl::queue &queue, int m, int n = 1, int val = 0){
    
    double *y = sycl::malloc_shared<double>(m*n, queue);
    for(int i = 0; i < m*n; i++)
        y[i] = val;

    return y;
}

       
// # hx function definition
double hx(double x1, double x2){
    
    return sqrt(x1*x1 + x2*x2);
}


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
    
double *cholesky(sycl::queue &queue, double *L, int64_t nb,double alpha){
/* # ====================
    
    #U    -> matriz Input
    #nb   -> numero de linhas do bloco U
    #alpha-> multiplicador escalar na função de Cholesky
    
   # ====================*/
    
    // # Make a copy of P (P must remain unchanged)
    double *U = sycl::malloc_shared<double>(nb*nb, queue);
    zero(U,nb,nb);
    
    auto copy_L = blas::row_major::copy(queue, nb*nb, 
                                        L, 1, U, 1);
    copy_L.wait();
    
    std::int64_t scratchpad_size = lapack::potrf_scratchpad_size<double>(queue, lowerM, nb, nb);
    double *scratchpad = sycl::malloc_shared<double>(scratchpad_size, queue);
    
    
    auto event1 = blas::scal(queue, nb*nb, alpha, U, 1.0);
    event1.wait_and_throw();
    
    
    auto event2 = lapack::potrf(queue, lowerM, nb, U, nb, scratchpad, scratchpad_size );
    event2.wait_and_throw();
    
    for(int i = 0; i<nb; i++){
        for(int j = 0; j<nb;j++){
            if(i>j) U[i*nb+j] = 0;
        }
    }
    
    free(scratchpad, queue);
    
    return U;
}




class RadarUKF{
public: 


    
    //# Setters member-functions 
    void setSttVariable(double * xset);
    void setTransitionCovMatrix(double * Qset);
    void setMeasureCovMatrix(double * Rset);
    void setErrorCovMatrix(double * Pset);
    
    //# getter memb. functions
    double getResult( int );
        
    
    //# prediction functions
    double *calc_weights();
    void SigmaPoints(double**);
    
    double *fx(double *,double dt);
    double *hx(double *);
    
    
    void UT_Hx(double **Hx_pts, double **sigma);
    void UT_Fx(double **Fx_pts, double **sigma, double dt);
    void UT(const double **sigma, int n, double* Noise, double* xm, double* cov);
        
    
    //# Update step
    void UKF( double *,double);
    
    void end_task();
private: 
    sycl::queue queue;
    
    //# Matrix Pointers
    double *Q;
    double *R;
    double *x;
    double *P; 
    double *z;
    
    
    //#Config. values
    double alpha = 1.0; 
    double beta  = 0.0;
    const int M = 3; 
    const int L = 1;

    const int WSize = M*2+1; //## Size of SigmaPoints Array
    double kappa = 0; 
    
    
    
    
    //# Sigma Points Allocation
    double *sigma_pts = sycl::malloc_shared<double>(7 * 3, queue);
    
    
    //# Weights to sigma_pts
    double *W = calc_weights();
    
    
    //# Intermediary values to calculus
    
    //#  fx function
    
    
    //# Sigma Point Arrays
    double **Xi  = sycl::malloc_shared<double*>(WSize, queue);
    double **fXi = sycl::malloc_shared<double*>(WSize, queue);
    double **hXi = sycl::malloc_shared<double*>(WSize, queue);
    

    //# fx_sigma Points function allocations: 
    
    

    
    //# calc_UT function allocations: 
    //#double *xm     = sycl::malloc_shared<double>(M * 1, queue);
    
    
    
    
    double *xcov = sycl::malloc_shared<double>(M * M, queue);
    
    
    
    
    
    //# UKF function allocations:
    double *zp    = sycl::malloc_shared<double>(L * L, queue);
    double *xp    = sycl::malloc_shared<double>(M * L, queue);
    
    double *Pz    = sycl::malloc_shared<double>(M * L, queue);
    double *Pxz   = sycl::malloc_shared<double>(M * L, queue);
    
    // # Array's Matrix
    

    
    
    double *KPz   = sycl::malloc_shared<double>(M * M, queue);
    double *Pp    = sycl::malloc_shared<double>(M * M, queue);
    
    double *K     = sycl::malloc_shared<double>(M * M, queue);
    double *AP    = sycl::malloc_shared<double>(M * M, queue);   
    
    
};

    //#setters functions
void RadarUKF::setSttVariable(double * xset){
    x = xset;
}

void RadarUKF::setTransitionCovMatrix(double * Qset){
    Q = Qset;
}

void RadarUKF::setMeasureCovMatrix(double * Rset){
    R = Rset;
}

void RadarUKF::setErrorCovMatrix(double * Pset){
    P = Pset;
}
        
    //# getter function
double RadarUKF::getResult(int l){
    return x[l];
}
    


//Weight Sigma-Points Calculus

double *RadarUKF::calc_weights(){
    int den = this->M + this->kappa;
    int size = this->M*2+1;
    double *W_set = sycl::malloc_shared<double>(size, queue);
    
    //set all values in W = 1/(2M+kappa)
    auto e1 = queue.parallel_for(sycl::range<1>(size-1), [=](id<1> i){W_set[i+1] = 1./(2*den);});
    e1.wait();             
    // Except W[0]
    W_set[0] = this->kappa / den;
    
    return W_set; 
}
    
double *RadarUKF::fx(double *x_in, double dt){
    
    double *fx_vec = sycl::malloc_shared<double>(M * 1, queue);
    double *A = sycl::malloc_shared<double>(M * M, queue);
    
    // # to-do:improve by using the Xi pointers and A Matrix (gemm_batch)
    
    
    // # set A matrix
    eye(A, M, 1.0);
    A[1] = dt;
    
    
    // # Fx = A*Xi
    auto fx_event = blas::row_major::gemm(queue, nontransM, nontransM,
                                          M, L, M, 
                                          1.0, A, M, 
                                          x_in, L, beta, fx_vec, L);
    fx_event.wait();
    free(A, queue);
    
    return fx_vec;
}
    
double *RadarUKF::hx(double *x_in){
    
    double x1 = x_in[0]; 
    double x2 = x_in[2]; 
    
    double *hx_vec = sycl::malloc_shared<double>(1 * 1, queue);
    hx_vec[0] = sqrt(x1*x1 + x2*x2);
    
    return hx_vec;
}
    

void RadarUKF::UT_Hx(double **Hx_pts, double **sigma_h){
    
    
    //## to-do: Parallel-for(?)

    for(int i = 0; i < WSize; i++){
        Hx_pts[i] = hx(sigma_h[i]);

    }
    
}

void RadarUKF::UT_Fx(double **Fx_pts, double **sigma, double dt){

    
    //## to-do: Parallel-for(?)
    for(int i=0; i< WSize;i++){
        Fx_pts[i] = fx(sigma[i], dt);

    }
}
void RadarUKF::SigmaPoints(double **SigmaPts){
    
    //# Declaring initial variables
    double scal = M + this->kappa;   
    
    
    // #Cholesky decomposition: fix-me ?
    auto U = cholesky(queue, P, M, scal);
    
    
    double **arr_1 = sycl::malloc_shared<double*>(WSize*2, queue);
    double **arr_2 = sycl::malloc_shared<double*>(WSize*2, queue);

    
    
    //#fix: copy_batch?
    for(int i = 0; i < WSize; i++){
        SigmaPts[i] = NullVector(queue, M);
        arr_1[WSize+i] = x;
        arr_2[i] = SigmaPts[i];
        arr_2[WSize+i] = SigmaPts[i];
    }
        
    
    //# Decompose U matrix in M vectors, to use in axpy_batch
    arr_1[0] = NullVector(queue, M);
    for(int i=0; i<M; i++){
        arr_1[i+1] = matrix2vector(queue, U, M, M, i);
        arr_1[M+i+1] = matrix2vector(queue, U, M, M, i);        
    }
  
    
    //## axpy_batch application: 
    //# sigmaPts[0] = x
    //# sigmaPts[i] = x +  1 *U[i]  , with i = 1,2, ..., (M-1)
    //# sigmaPts[i] = x +(-1)*U[i]  , with i = M, M+1, ..., 2M. 

    //# set data to axpy_batch: cov = cov+scal(W[i])*(Y x Y.t)
    //# value function axpy_batch
    const int GRP = 4;                                       //# Total de grupos
    const std::int64_t n_axpy[GRP] = {M, M, M, M};           //# Total de elementos por operação
    double a_axpy[GRP]             = {0.0, 1.0, -1.0, 1.0};  //# Escalar da operação
    const std::int64_t incr[GRP]   = {1, 1, 1, 1};           //# Distancia entre os elementos
    std::int64_t sz_axpy[GRP]      = {1, M, M, WSize};       //# Qtd. soma/subtracao por grupo
        
    
    auto sigma_ev = blas::axpy_batch(queue, n_axpy, a_axpy, 
                                     (const double **) arr_1, incr,
                                     arr_2, incr, 
                                     GRP, sz_axpy);
 
    sigma_ev.wait();            
    
    free(arr_1,queue);
    free(arr_2,queue);
}
    
void RadarUKF::UT(const double **sigma, int n, double* Noise, double* xm, double* xcov){
    
    // ## Set xm: xm += scal(W[i])*sigma[i]   ,  with i=0, 1, ..., WSize.
    // #Variables to axpy_batch:
    double **xm_arr = sycl::malloc_shared<double*>(WSize, queue);
    
    const int xm_grp = WSize; 
    std::int64_t n_scal[xm_grp] ;
    double a_scal[xm_grp];
    std::int64_t incr_scal[xm_grp];
    std::int64_t sz_scal[xm_grp];
    
    
    //#xm = NullVector(queue, n);
    
    for(int i=0; i< WSize; i++){
        n_scal[i]    = n;
        a_scal[i]    = W[i];
        incr_scal[i] = 1; 
        sz_scal[i]   = 1;
        xm_arr[i] = xm;
    }

    // # xm = xm + scal(W[i])*sigma[i]
    auto scal_event = blas::axpy_batch(queue, n_scal, a_scal, 
                                      sigma, incr_scal,
                                      xm_arr, incr_scal, 
                                      xm_grp, sz_scal);
    scal_event.wait();

    
    //# fix me: tamanho de Y
    std::int64_t copy_sz[1]   = {n};
    std::int64_t copy_incr[1] = {1};
    std::int64_t copy_grp[1]  = {WSize};
    
    double **Y = sycl::malloc_shared<double*>(WSize, queue);

    //# Fill Y vector with NullVectors
    for(int i = 0; i < WSize;i++){
        Y[i]      = NullVector(queue, n);
    }
    
    auto copy_event = blas::row_major::copy_batch(queue, copy_sz,
                                                  sigma, copy_incr,
                                                  Y, copy_incr, 1, copy_grp);
    copy_event.wait();
    
    
     //### 1.0) Y[i] = - xm + Y[i]  == -xm + sigma[i] 
    // ## Initial conditions for axpy_batch: Y[i] = sigma[i] - xm    
    
    
    const std::int64_t s_axpy[] = {n};
    double a_axpy[]             = {-1.0}; 
    const std::int64_t s_incr[] = {1};
    std::int64_t sz_axpy[]      = {WSize};
    

    auto sum_event = blas::axpy_batch(queue, s_axpy, a_axpy, 
                                      (const double **) xm_arr, s_incr,
                                      Y, s_incr, 
                                      1, sz_axpy);
    sum_event.wait(); 
    

    
    //## Covariation Calculus: cov = Noise + sum(W[i] * (Y[i] x Y[i].T),     with i = 0, 1, ..., n. 
    //# 1.1)  cov = cov+scal(W[i])*(Y x Y.t) 
    
    //# set data to gemm_batch: cov = cov+scal(W[i])*(Y x Y.t)      
    //# value function gemm_batch
    double **cov_arr = sycl::malloc_shared<double*>(WSize, queue); 
    
    const int cov_grp = WSize; 
    oneapi::mkl::transpose transA[cov_grp];
    oneapi::mkl::transpose transB[cov_grp];
    
    std::int64_t m_lst[cov_grp];
    std::int64_t k_lst[cov_grp];
    
    std::int64_t lda_lst[cov_grp];
    std::int64_t ldc_lst[cov_grp] ;
    
    double a_lst[cov_grp];
    double b_lst[cov_grp];
    
    std::int64_t grp_sz[cov_grp];
    
    
    for(int it=0; it < cov_grp; it++){
        transA[it]  = nontransM;
        transB[it]  = transM;
        m_lst[it]   = n; 
        k_lst[it]   = 1; 
        lda_lst[it] = 1;
        ldc_lst[it] = n;
        a_lst[it]   = W[it]; 
        b_lst[it]   = 1.0;
        grp_sz[it]  = 1; 
        // # set C array with cov adress
        cov_arr[it]  = xcov;
        
    }

    
    // # cov = cov + scalar(W[i]) *(Y[i] x Y[i].T)
    auto gemm_event = blas::row_major::gemm_batch(queue,
                                                  transA, transB,
                                                  m_lst, m_lst, k_lst, a_lst, 
                                                  (const double **) Y, lda_lst, 
                                                  (const double **) Y, lda_lst, 
                                                  b_lst, cov_arr, ldc_lst, 
                                                  cov_grp, grp_sz);
    
    gemm_event.wait();
    
    
    
    //# 1.2) cov = cov + Noise

    
    
    auto sum_event2 = blas::axpy(queue, n*n, alpha, Noise , 1.0, xcov, 1.0);
    sum_event2.wait(); 
    
    
    // # Free memory for auxiliary pointers.
    free(cov_arr, queue);
    free(xm_arr, queue);
    free(Y, queue);
}

void RadarUKF::UKF(double *z, double dt){
    
    
    SigmaPoints(Xi);
    
    UT_Fx(fXi, Xi, dt);
    UT_Hx(hXi, Xi);  
    
    zero(xp, M, L);
    zero(Pp,M,M);
    UT((const double **)fXi, M, Q, xp, Pp);
    
    zero(zp, L, L);
    zero(Pz, L, L);
    UT((const double **)hXi, L, R, zp, Pz);
    
    // ### Pxz = scal(W[i]) * ( {fXi[i]- xp}x{hXi[i]- zp}.T),    with i = 0, 1, ...., n.  
    // ## 1.0) Auxilar Vectors:
    // # a[i] = fXi[i] - xp
    // # b[i] = hXi[i] - zp
    
    double **a       = sycl::malloc_shared<double*>(WSize, queue);
    double **b       = sycl::malloc_shared<double*>(WSize, queue);
    double **xp_arr  = sycl::malloc_shared<double*>(WSize, queue);
    double **zp_arr  = sycl::malloc_shared<double*>(WSize, queue);
    
    double **Pxz_arr = sycl::malloc_shared<double*>(WSize, queue);
    double *Pz_copy  = sycl::malloc_shared<double>(M * L, queue);
    
    const int GRP = 1; 
    const std::int64_t fxi_axpy[GRP] = {M};
    const std::int64_t hxi_axpy[GRP] = {L};
    double a_axpy[GRP] = {-1.0};
    
    const std::int64_t incr[GRP] = {1};
    
    std::int64_t sz_axpy[GRP] = {WSize};
    
    
    for(int it=0; it < WSize; it++){
        xp_arr[it] = xp;
        zp_arr[it] = zp;
    }

    // # fXi[i] = (-)xp + fXi[i]
    auto ev1 = blas::axpy_batch(queue, fxi_axpy, a_axpy, 
                                      (const double **) xp_arr, incr,
                                      fXi, incr, 
                                      GRP, sz_axpy);
    ev1.wait();  
    
    // # hXi[i] = (-)zp + hXi[i]
    auto ev2 = blas::axpy_batch(queue, hxi_axpy, a_axpy, 
                                      (const double **) zp_arr, incr,
                                      hXi, incr, 
                                      GRP, sz_axpy);
    ev2.wait();  
    

    // ## 1.1) outer product and scalar: W[i] * ({a[i]} x {b[i]}.T)
    
    
    //# Definitions to gemm_batch: Pxz
    const int grp_Pxz = WSize; 
    oneapi::mkl::transpose transA[grp_Pxz];
    oneapi::mkl::transpose transB[grp_Pxz];
    std::int64_t m_lst[grp_Pxz];
    std::int64_t k_lst[grp_Pxz];
    std::int64_t lda_lst[grp_Pxz];
    std::int64_t ldc_lst[grp_Pxz];
    double a_lst[grp_Pxz];
    double b_lst[grp_Pxz];
    std::int64_t grp_sz[grp_Pxz];
    
    // # Array pointer to Pxz
    
    zero(Pxz, M, M); 

    
    for(int it=0; it < WSize; it++){
        transA[it]  = nontransM;
        transB[it]  = nontransM;
        m_lst[it]   = M;
        k_lst[it]   = 1; 
        lda_lst[it] = 1;
        ldc_lst[it] = M;
        a_lst[it]   = W[it];
        b_lst[it]   = 1.0;
        grp_sz[it]  = 1; 
        
        // # set Pxz_arr
        Pxz_arr[it]  = Pxz;
    }
    
    

    
    auto Pxz_event = blas::row_major::gemm_batch(queue,
                                                 transA, transB,
                                                 m_lst, k_lst, k_lst, a_lst,
                                                 (const double **) fXi, lda_lst, 
                                                 (const double **) hXi, lda_lst, 
                                                 b_lst, Pxz_arr, lda_lst, 
                                                 grp_Pxz, grp_sz);
    Pxz_event.wait();
      
    

    
    
    //## ====================================================== Aqui para baixo, esta OK ===================================================================
    //## K  = Pxz * inv(Pz)
    
    // # Pz = inv(Pz)
    // # OBS: make a copy of Pz, will be used on P calculus
    auto Pz_event = blas::row_major::copy(queue, M*L, 
                          Pz, 1, Pz_copy, 1);
    Pz_event.wait();
    inv(queue, Pz, L);
    
    
    // # K = Pxz * Pz
    auto K_event = blas::row_major::gemm(queue, nontransM, nontransM, M, L, L, alpha, Pxz, L, Pz, L, beta, K, L);
    K_event.wait();
    
    //## x = xp + K * (z-zp)
    // # z = z-zp
    auto zzp_event = blas::axpy(queue, L*L, -alpha, zp, 1.0, z, 1.0);
    zzp_event.wait();

    // # x = K*z -> K*(z-zp)
    auto Kz_event = blas::row_major::gemm(queue, nontransM, nontransM, M, L, L, alpha, K, L, z, L, beta, x, L);
    Kz_event.wait(); 
    
    // # x = xp + x
    auto x_event = blas::axpy(queue, M*L, alpha, xp, 1.0, x, 1.0);
    x_event.wait();
    
    
    //## P = Pp - K*(cp_Pz)*K_t
    // # KPz = K*Pz
    auto KPz_event = blas::row_major::gemm(queue, nontransM, nontransM,
                                           M, L, L,
                                           alpha, K, L,
                                           Pz_copy, L, beta,
                                           KPz, L);
    KPz_event.wait();
    zero(P,M,M);
    // # P = (-)KPz*K_t
    auto P1_event = blas::row_major::gemm(queue, nontransM, transM,
                                           M, M, L,
                                           -alpha, KPz, L,
                                           K, L, beta,
                                           P, M);
    P1_event.wait();
        
    // # P = Pp+(-)P
    auto P2_event = blas::axpy(queue, M*M, alpha, Pp, 1.0, P, 1.0);
    P2_event.wait();
    //# End calculus here, then its necessary to acess it by GetResult, 
    //# which is obtained by the matrix X(nRow x nCol).  
    
        
}

/*
void RadarUKF::~RadarUKF(){
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
    free(Kz, queue);
    
}
*/

