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
#include <stdlib.h>
#include <stdio.h>
#include <limits>
#include <random>
//queue q{property::queue::in_order()};
#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"
namespace blas = oneapi::mkl::blas;
namespace lapack = oneapi::mkl::lapack;

void GetVolt(float *Z) {
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0.0, 1.0);
	float w = 0 + 4*distribution(generator);
	*Z = 14.4 + w;
	//return z;
}
//matrix inverse
void inv(sycl::queue &queue, float *A, int64_t N);
//identity matrix
void eye(size_t N,float *A, float alpha);
//display matrix
void display(size_t rF, size_t cS, float *matrix);
//double SimpleKalman(double z);
void zero(size_t n, size_t m, float *C);

int main(){
	auto async_handler = [](sycl::exception_list exceptions) {
			for (std::exception_ptr const &e : exceptions) {
				try {
					std::rethrow_exception(e);
				}
				catch (sycl::exception const &e) {
					std::cout << "Caught asynchronous SYCL exception: " << e.what() << std::endl;
				}
			}
		};
	try {
		//propriedades matriz
		constexpr int N = 1;


		auto nontransM = oneapi::mkl::transpose::nontrans;
		auto transM = oneapi::mkl::transpose::trans;
		float alpha = 1.0; float beta = 0.0;


		sycl::device device = sycl::device(sycl::default_selector());
		std::cout << "Device: " << device.get_info<sycl::info::device::name>() << "\n";
		sycl::queue queue(device, async_handler);
		//sycl::queue queue{property::queue::in_order()};

        std::vector<sycl::event> event_list;

		double dt = 0.2, first = 0.0, volt;
		int Nsamples = 1;
		double t[Nsamples];

		auto *Xsaved = sycl::malloc_shared<double>(Nsamples, queue);
		auto *Zsaved = sycl::malloc_shared<double>(Nsamples, queue);


		//Alocação & valores iniciais
		float *A = sycl::malloc_shared<float>(1, queue); A[0] = 1.0;
		float *H = sycl::malloc_shared<float>(1, queue); H[0] = 1.0;
		float *Q = sycl::malloc_shared<float>(1, queue); Q[0] = 0.0;
		float *R = sycl::malloc_shared<float>(1, queue); R[0] = 4.0;
		float *x = sycl::malloc_shared<float>(1, queue); x[0] = 14.0;
		float *P = sycl::malloc_shared<float>(1, queue); P[0] = 6.0;
        
        float *z = sycl::malloc_shared<float>(1, queue);      
        
        

		//Memoria alocada para operações matriciais
		float *xp = sycl::malloc_shared<float>(1, queue);
		float *Pp = sycl::malloc_shared<float>(1, queue);
		float *K = sycl::malloc_shared<float>(1, queue);
		float *AP = sycl::malloc_shared<float>(1, queue);
		float *PpHT = sycl::malloc_shared<float>(1, queue);
		float *HpHTR = sycl::malloc_shared<float>(2, queue);
		float *Hxp = sycl::malloc_shared<float>(1, queue);
		float *Kz = sycl::malloc_shared<float>(1, queue);
		float *KH = sycl::malloc_shared<float>(1, queue);
        

		for (int i = 0; i < Nsamples; i++) {
			constexpr int gemm_total = 9, axpy_total = 4;
			sycl::event gemm_task[gemm_total], axpy_task;
            sycl::event axpy_task[axpy_total];
			std::vector<sycl::event> gemm[gemm_total];
            
            //t[i] = first += dt;
			//A[1]+=dt;
			// GetVolt(z);
			//Zsaved[i] = z;
            
            
            // xp = A * x // OK 
			gemm_task[0] = blas::gemm(queue, nontransM, nontransM, 1, 1, 1, alpha, A, 1, x, 1, beta, xp, 1, gemm[0]);
			gemm_task[0].wait();
            
			// Pp = A * P * A' + Q
				//1.1) AP = A * P
			gemm_task[1] = blas::gemm(queue, nontransM, nontransM, 1, 1, 1, alpha, A, 1, P, 1, beta, AP, 1,gemm[1]);
			gemm_task[1].wait(); //ok
             
            
				//1.2) Pp = AP * A'
			gemm_task[2] = blas::gemm(queue, nontransM, transM, 1, 1, 1, alpha, AP, 1, A, 1, beta, Pp, 1, gemm[2]);
			gemm_task[2].wait();
            
            
                //1.3)Pp = Pp + Q
            axpy_task[0] = blas::axpy(queue, 1, alpha , Q , 1.0, Pp, 1.0);
            axpy_task[0].wait();
            
            // K = Pp * H' * inv(H * Pp * H' + R)
				//2.1) PpHT = Pp * H' -->  dimensao PpHT: (M * N)
			gemm_task[3] = blas::gemm(queue, nontransM, transM, 1, 1, 1, alpha, Pp, 1, H, 1, beta, PpHT, 1, gemm[3]);
			gemm_task[3].wait();
                //2.2) HpHTR = H * (Pp * H') = H (NxM) * PpHT(MxN) --> HpHTR (NxN)
			gemm_task[4] = blas::gemm(queue, nontransM, nontransM, 1, 1, 1, alpha, H, 1, PpHT, 1, beta, HpHTR, 1,gemm[4]);
			gemm_task[4].wait();
				// 2.3) HpHTR = HpHTR + R
            axpy_task[1] = blas::axpy(queue, 1, alpha, R, 1.0, HpHTR, 1.0);
			axpy_task[1].wait();
          
                //HpHTR = inv(HpHTR)
            inv(queue, HpHTR, 1);
            
            // 2.4) K = (Pp * H') * HpHTR ==> PpHT(MxN) * HpHTR(NxN) ---> K(MxN)
            gemm_task[5] = blas::gemm(queue, nontransM, nontransM, 1, 1, 1, alpha, PpHT, 1, HpHTR, 1, beta, K, 1,gemm[5]);
            gemm_task[5].wait();
           // x = xp + K * (z - H * xp)          
                //3.1) Hxp = H(NxM) * xp(MxN) --> Hxp(NxN)

            gemm_task[6] = blas::gemm(queue, nontransM, nontransM, 1, 1, 1, alpha, H, 1, xp, 1, beta, Hxp, 1, gemm[6]);
			gemm_task[6].wait();
            	//3.2) z = -Hxp(NxN) + z(NxN)
			axpy_task[2] = blas::axpy(queue, 1, -alpha, Hxp, 1.0, z, 1.0);
            axpy_task[2].wait();
            	//3.3) Kz = K*z --> Kz(MxN)
			gemm_task[7] = blas::gemm(queue, nontransM, nontransM, 1, 1, 1, alpha, K, 1, z, 1, beta, Kz, 1, gemm[7]);
            gemm_task[7].wait();
            
            std::cout<<"resultado xp: "<<std::endl; 
            std::cout<<xp[0]<<std::endl;
            std::cout<<"resultado Kz: "<<std::endl; 
            std::cout<<Kz[0]<<std::endl;
                //3.4) xp = xp + Kz
			axpy_task[3] = blas::axpy(queue, 1, alpha, Kz, 1.0, xp, 1.0);
            axpy_task[3].wait();
            // P = Pp - K * H * Pp
        std::cout<<"resultado xp: "<<std::endl; 
            std::cout<<xp[0]<<std::endl;
				//4.1) KH = K(MxN)*H(NxM) 
            gemm_task[8] = blas::gemm(queue, nontransM, nontransM, 1, 1, 1, alpha, K, 1, H, 1, beta, KH, 1, gemm[8]);
            gemm_task[8].wait();
                //4.2) P = KH(MxM) * Pp(MxM)
            
			//gemm_task[9] = blas::gemm(queue, nontransM, nontransM, 1, 1, 1, alpha, KH, 1, Pp,1, beta, P, 1, gemm[9]);
            //gemm_task[9].wait();
            	//4.3) P = (-Pp + P)
            
			//axpy_task[4] = blas::axpy(queue, 1, -alpha, Pp, 1.0, P, 1.0);
            //axpy_task[4].wait();
                //4.4) P = -P
            //scal_task = blas::scal(queue, 1, -alpha, P, 1.0);
            
        }
		if (!Xsaved || !Zsaved) {
			std::cerr << "Could not allocate memory for vectors." << std::endl;
			exit(1);
		}
		//queue.wait_and_throw();

		free(x, queue);
		free(A, queue);
		free(xp, queue);
		/*liberação da memória após finalizaçao do programa.
		free(Xsaved, queue);
		free(Zsaved, queue);

		free(H, queue);
		free(Q, queue);
		free(R, queue);

		free(P, queue);

		free(Pp, queue);
		free(K, queue);
		free(AP, queue);
		free(PpHT, queue);
		free(HpHTR, queue);
		free(Hxp, queue);
		free(Kz, queue);
		free(KH, queue);*/


	} catch (const std::exception &e) {
		std::cerr << "An exception occurred: "
				  << e.what() << std::endl;
		exit(1);
	}
}

void display(size_t rowFirst, size_t columnSecond, float *mult){
	std::cout << "Output Matrix:" << std::endl;
	for(int i = 0; i < rowFirst*columnSecond; ++i){
		if(i % columnSecond==0){
			std::cout << std::endl << std::endl;
		}
		std::cout << mult[i] <<" ";
	}
    std::cout<<"\n";
}

void eye(size_t N, float *P, float alpha){
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			if (i == j) P[i*N+j] = 1.0*alpha;
			else P[i*N+j] = 0.0;
		}
	}
}
void zero(size_t n, size_t m, float *C){
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			C[i*m+j] = 0.0;
		}
	}
}
void inv(sycl::queue &queue, float *A, int64_t N){
//modulo de procedimento de inversa da matriz, incluindo a fatoração.
// A          ==> Matriz a ser inversa (referencia)
// N          ==> dimensão da matriz (C.E: A(l x c), sendo l=c)
// scratchpad ==> vetor a ser usado na operação de inversao (shared)
// scrat_size ==> total da memória alocada para operação
    sycl::event getr_task[2];
    std::vector<sycl::event> event_list;
    //oneapi::mkl::lapack::getrf(q, n, n, A.data(), n, piv.data(), work.data(), iwork);

    // Scratchpad & Scratch_size;
    float scratch_size = lapack::getrf_scratchpad_size<float>(queue, N, N, N);
    float *scratchpad = sycl::malloc_shared<float>(scratch_size+1, queue);


    //IPIV
    auto *IPIV = sycl::malloc_shared<int64_t>(N*N, queue);
    
    
    getr_task[0] = lapack::getrf(queue, N, N, A, N, IPIV, scratchpad, scratch_size, event_list);
    getr_task[0].wait();
    getr_task[1] = lapack::getri(queue, N, A, N, IPIV, scratchpad, scratch_size, event_list);
    getr_task[1].wait();
    
    //delete IPIV;
}
