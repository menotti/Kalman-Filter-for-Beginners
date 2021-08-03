//==============================================================
// Copyright © 2020 Intel Corporation
//
// SPDX-License-Identifier: MIT
// Quick list & links
//gemm(queue, transA, transB, l_A, c_A, c_B, alpha, A_usm, ldA, B_usm, ldB, beta, C_usm, ldC, gemm_dependencies);
//getrf(queue, m, n, A, lda, ipiv, scratchpad_dev, scratchpad_size,sycl::vector_class<sycl::event>{ in_event });
//getri(queue, n, A_dev, lda, ipiv_dev, scratchpad, scratchpad);
//scal(queue, n, scalar, vector, incx)
//https://stackoverflow.com/questions/3519959/computing-the-inverse-of-a-matrix-using-lapack-in-c
// =============================================================
#include <iostream>
#include <limits>
#include <random>

#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"

namespace blas_lib = oneapi::mkl::blas;
namespace lap_lib = oneapi::mkl::lapack;



/*        std::vector<int64_t> ipiv(1);
        std::allocator<cl::sycl::event> dependencies;*/


double GetVolt() {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);
    double w = 0 + 4*distribution(generator);
    double z = 14.4 + w;
    return z;
}
//identity matrix
void eye(double &P,const int N, double alpha);
void mt_print(double *A,int m,int n);
void inverse(sycl::queue &dev_queue, double A, std::int64_t N){
//modulo de procedimento de inversa da matriz, incluindo a fatoração.
// A          ==> Matriz a ser inversa (referencia)
// N          ==> dimensão da matriz (C.E: A(l x c), sendo l=c)
// scratchpad ==> vetor a ser usado na operação de inversçao (shared)
// scrat_size ==> total da memória alocada para operação


//double SimpleKalman(double z);

int main(){

    try {
        //matrix dimensions & properties
        int *M = new int(2);
        int *N = new int(1);
        const double alpha = 1.0;
        const double beta = 0.0;
        auto nontransM = oneapi::mkl::transpose::nontrans;
        auto transM = oneapi::mkl::transpose::trans;
        
        sycl::queue device_queue{sycl::default_selector{}};

        std::cout << "Device: "
                  << device_queue.get_device().get_info<sycl::info::device::name>()
                  << std::endl;

        double dt = 0.2, first = 0.0, volt;
        int Nsamples = 10;
        double t[Nsamples];
        //Memory allocation & matrix definition
        auto Xsaved = sycl::malloc_shared<double>(Nsamples, device_queue);
        auto Zsaved = sycl::malloc_shared<double>(Nsamples, device_queue);

        auto A = sycl::malloc_shared<double>(M*M, device_queue);
        eye(A, M, 1);      
        //auto H = sycl::malloc_shared<double>(1, device_queue);
        //H[0] = 1.0;
        auto Q = sycl::malloc_shared<double>(M*M, device_queue);
        Q[0][0] = 1;Q[0][1] = 0;
        Q[1][2] = 0;Q[1][3] = 3;
        //auto R = sycl::malloc_shared<double>(1, device_queue);
        //R[0] = 4.0;
        auto x = sycl::malloc_shared<double>(N*K, device_queue);
        x[0][0] = 0.0; x[0][1] = 20.0;
        auto P = sycl::malloc_shared<double>(M*M, device_queue);
        eye(P,M,5);
        
        auto z = sycl::malloc_shared<double>(1, device_queue);*/
        
        auto xp = sycl::malloc_shared<double>(M*N, device_queue);/*
        
        auto Pp = sycl::malloc_shared<double>(1, device_queue);
        auto K  = sycl::malloc_shared<double>(1, device_queue);
        auto AP = sycl::malloc_shared<double>(1, device_queue);
        auto PpHT = sycl::malloc_shared<double>(1, device_queue);
        auto HpHTR = sycl::malloc_shared<double>(1, device_queue);
        auto Hxp   = sycl::malloc_shared<double>(1, device_queue);
        auto Kz    = sycl::malloc_shared<double>(1, device_queue);
        auto KH    = sycl::malloc_shared<double>(1, device_queue);*/

        //Matrix Dimensions:
        const int M = 1;
        const int N = 1;
        const int  = 1;
        for (int i = 0; i < Nsamples; i++) {
        	cl::sycl::vector_class<sycl::event> event;
        	 const sycl::vector_class<sycl::event> &dependencies = {};
            t[i] = first += dt;
            Zsaved[i] = GetVolt();
            z = &Zsaved[i];

            // xp = A * x
            blas_lib::gemm(device_queue, nontransM, nontransM, N, N, K, alpha, A, 1, x, 1, beta, xp, 1);
            
            // Pp = A * P * A' + Q (função se mantém constante)
            	// AP = A * P
           /* blas_lib::gemm(device_queue, nontransM, nontransM, 1, 1,1, alpha, A,1 , P, 1, beta, AP, 1);
              	// Pp = AP * A'
            blas_lib::gemm(device_queue, nontransM, transM, 1, 1, 1, alpha, AP, 1, A, 1, beta, Pp, 1);
            	// Pp = Pp + Q
            blas_lib::axpy(device_queue, 1, 1, Q, 1, Pp, 1);
            // K = Pp * H' * inv(H * Pp * H' + R)
              	  // PpHT = Pp * H'
            blas_lib::gemm(device_queue, nontransM, transM, 1, 1, 1, alpha, Pp, 1, H, 1, beta, PpHT, 1);
                  // HpHTR = H * (Pp * H') = H * PpHT
            blas_lib::gemm(device_queue, nontransM, nontransM, 1, 1, 1, alpha, PpHT, 1, H, 1, beta, HpHTR, 1);
                  // HpHTR = HpHTR + R
            blas_lib::axpy(device_queue, 1, 1, R, 1, HpHTR, 1);
                  // HpHTR = inv(HpHTR)
            inverse(device_queue, *HpHTR, 1);
              //K = (Pp * H') * HpHTR ==> PpHT * HpHTR (função se mantém constante)
            blas_lib::gemm(device_queue, nontransM, nontransM, 1, 1, 1, alpha, PpHT, 1, HpHTR, 1, beta, K, 1);

            // x = xp + K * (z - H * xp)
                //Hxp = H * xp
            blas_lib::gemm(device_queue, nontransM, nontransM, 1, 1, 1, alpha, H, 1, xp, 1, beta, Hxp, 1);
                //z = -Hxp + z
            blas_lib::axpy(device_queue, 1, (-1)*alpha, Hxp, 1, z, 1);

                //Kz = K*z
            blas_lib::gemm(device_queue, nontransM, nontransM, 1, 1, 1, alpha, K, 1, z, 1, beta, Kz, 1);
                // xp = xp + Kz
            blas_lib::axpy(device_queue, 1, alpha, Kz, 1, xp, 1);
            // P = Pp - K * H * Pp
                //KH = K*H
            blas_lib::gemm(device_queue, nontransM, nontransM, 1, 1, 1, alpha, K, 1, H, 1, beta, KH, 1);
                //P = KH * Pp
            blas_lib::gemm(device_queue, nontransM, nontransM, 1, 1, 1, alpha, KH, 1, Pp, 1, beta, P, 1);
                // P = -Pp + P
            blas_lib::axpy(device_queue, 1, -alpha, Pp, 1, P, 1);
            // P = -P (Inversão da operação anterior)
            //scal(queue, n, scalar, vector, incx)
            blas_lib::scal(device_queue, 1, -alpha, P, 1);/*/

            // volt = x
            Xsaved[i] = volt;
        }


        if (!Xsaved || !Zsaved) {
            std::cerr << "Could not allocate memory for vectors." << std::endl;
            exit(1);
        }
        device_queue.wait_and_throw();

        //liberação da memória após finalizaçao do programa.
       /* free(Xsaved, device_queue);
        free(Zsaved, device_queue);
        free(A, device_queue);
        free(H, device_queue);
        free(Q, device_queue);
        free(R, device_queue);
        free(x, device_queue);
        free(P, device_queue);
        free(xp, device_queue);
        free(Pp, device_queue);
        free(K, device_queue);
        free(AP, device_queue);
        free(PpHT, device_queue);
        free(HpHTR, device_queue);
        free(Hxp, device_queue);
        free(Kz, device_queue);
        free(KH, device_queue);*/


    } catch (const std::exception &e) {
        std::cerr << "An exception occurred: "
                  << e.what() << std::endl;
        exit(1);
    }
}


