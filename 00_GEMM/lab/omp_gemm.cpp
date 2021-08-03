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

double GetVolt() {
	std::default_random_engine generator;
	std::normal_distribution<double> distribution(0.0, 1.0);
	double w = 0 + 4*distribution(generator);
	double z = 14.4 + w;
	return z;
}
/*void inverse(sycl::queue &dev_queue, double &A, int N, double &scratchpad, int scratch_size){
//modulo de procedimento de inversa da matriz, incluindo a fatoração.
// A          ==> Matriz a ser inversa (referencia)
// N          ==> dimensão da matriz (C.E: A(l x c), sendo l=c)
// scratchpad ==> vetor a ser usado na operação de inversçao (shared)
// scrat_size ==> total da memória alocada para operação
	int *IPIV = new int[N];
	lapack::getrf(dev_queue, N, N, A, N, IPIV, scratchpad, scratch_size);
	lapack::getri(dev_queue, N, A, N , IPIV, scratchpad,scratch_size);
	delete[] IPIV;
}*/
//identity matrix
void eye(size_t N,float *A, float alpha);
//show matrix
void display(size_t rF, size_t cS, float *matrix);
//multiply
void mult(float C[], float A[], float B[],int m,int n,int k);
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
		constexpr int M = 2;
		constexpr int N = 1;


		auto nontransM = oneapi::mkl::transpose::nontrans;
		auto transM = oneapi::mkl::transpose::trans;
		float alpha = 1.0; float beta = 1.0;


		sycl::device device = sycl::device(sycl::default_selector());
		std::cout << "Device: " << device.get_info<sycl::info::device::name>() << "\n";
		sycl::queue queue(device, async_handler);
		//sycl::queue queue{property::queue::in_order()};

		double dt = 0.2, first = 0.0, volt;
		int Nsamples = 1;
		double t[Nsamples];

		auto Xsaved = sycl::malloc_shared<double>(Nsamples, queue);
		auto Zsaved = sycl::malloc_shared<double>(Nsamples, queue);


		//Matrizes iniciais
		float *A = sycl::malloc_shared<float>(M*M, queue);
		eye(M, A, 1);
		float *H = sycl::malloc_shared<float>(M*N, queue);
		H[0] = 1; H[1] = 0;
		float *Q = sycl::malloc_shared<float>(M*M, queue);
		Q[0] = 1; Q[1] = 0;
		Q[2] = 0; Q[3] = 3;
		float *R = sycl::malloc_shared<float>(N*N, queue);
		R[0] = 10;
		float *x = sycl::malloc_shared<float>(M*N, queue);
		x[0] = 0; x[1] = 20;
		float *P = sycl::malloc_shared<float>(M*M, queue);
		eye(M, P, 5); //matriz ident. * 5



		//Memoria alocada para operações matriciais
		float *xp = sycl::malloc_shared<float>(M*N, queue);
		//float *Pp = sycl::malloc_shared<float>(M*M, queue);
		//float *K = sycl::malloc_shared<float>(M*N, queue);
		float *AP = sycl::malloc_shared<float>(M*M, queue);
		//float *PpHT = sycl::malloc_shared<float>(M*N, queue);
		//float *HpHTR = sycl::malloc_shared<float>(N*N, queue);
		//float *Hxp = sycl::malloc_shared<float>(1, queue);
		//float *Kz = sycl::malloc_shared<float>(1, queue);
		//float *KH = sycl::malloc_shared<float>(1, queue);


		for (int i = 0; i < Nsamples; i++) {
			//t[i] = first += dt;
			//A[1]+=dt;
			//Zsaved[i] = GetVolt();
			//z = Zsaved[i];

			// xp = A * x
			constexpr int total = 12;
			sycl::event order[total];
			std::vector<sycl::event> op_dep[total];
			order[0] = blas::gemm(queue, nontransM, nontransM, M, M, N, alpha, A, M, x, N, beta, xp, N,op_dep[0]);
			order[0].wait();
			//impressao do veotr
			for(int i =0; i<4;i++) std::cout<<xp[i] << " ";
			std::cout<<"\n";
			
			// Pp = A * P * A' + Q (função se mantém constante)
				// AP = A * P
			order[1] = blas::gemm(queue, nontransM, nontransM, M, M, M, alpha, A, M, P, M, beta, AP, M);
			order[1].wait();
			for(int i =0; i<4;i++) std::cout<<AP[i] << " ";
						std::cout<<"\n";
				// Pp = AP * A'
			/*order[2] = blas::gemm(queue, nontransM, transM, M, M, M, alpha, AP, M, A, M, beta, Pp, M);
			order[2].wait();
			for(int i =0; i<4;i++) std::cout<<A[i] << " ";
			std::cout<<"\n";
			for(int i =0; i<4;i++) std::cout<<Pp[i] << " ";
			std::cout<<"\n";

									
				// Pp = Pp + Q
			order[3] = blas::axpy(queue, M*M, alpha , Q, 1.0, Pp, 1.0);
			// K = Pp * H' * inv(H * Pp * H' + R)
			
				  // PpHT = Pp * H' -->  PpHT (M * N)
			order[4] = blas::gemm(queue, nontransM, transM, M, M, N, alpha, Pp, M, H, N, beta, PpHT, N);
				  // HpHTR = H * (Pp * H') = H * PpHT --> HpHTR (N * N)

			order[5] = blas::gemm(queue, nontransM, nontransM, N, M, N, alpha, H, M, PpHT, M, beta, HpHTR, N);
			display(N,N,HpHTR);
				// HpHTR = HpHTR + R
			order[6] = blas::axpy(queue, 1, 1, R, 1, HpHTR, 1);
				  // HpHTR = inv(HpHTR)
			inverse(queue, HpHTR, 1, scratchpad, scratchpad_size);
			  //K = (Pp * H') * HpHTR ==> PpHT * HpHTR (função se mantém constante)
			order[7] = blas::gemm(queue, nontransM, nontransM, 1, 1, 1, alpha, PpHT, 1, HpHTR, 1, beta, K, 1);

			// x = xp + K * (z - H * xp)
				//Hxp = H * xp
			order[8] = blas::gemm(queue, nontransM, nontransM, 1, 1, 1, alpha, H, 1, xp, 1, beta, Hxp, 1);
				//z = -Hxp + z
			lapack::axpy(queue, 1, -alpha, Hxp, 1, z, 1);
				//Kz = K*z
			order[9] = blas::gemm(queue, nontransM, nontransM, 1, 1, 1, alpha, K, 1, z, 1, beta, Kz, 1);
				// xp = xp + Kz
			lapack::axpy(queue, 1, alpha, Kz, 1, xp, 1);
			// P = Pp - K * H * Pp
				//KH = K*H
			order[10] blas::gemm(queue, nontransM, nontransM, 1, 1, 1, alpha, K, 1, H, 1, beta, KH, 1);
				//P = KH * Pp
			order[11] = blas::gemm(queue, nontransM, nontransM, 1, 1, 1, alpha, KH, 1, Pp, beta, P, 1);
				// P = -Pp + P
			blas::axpy(queue, 1, -alpha, Pp, 1, P, 1);
			// P = -P (Inversão da operação anterior)
			//scal(queue, n, scalar, vector, incx)
			blas::scal(queue, 1, -alpha, P, 1);

			// volt = x
			Xsaved[i] = volt;*/
		}


		if (!Xsaved || !Zsaved) {
			std::cerr << "Could not allocate memory for vectors." << std::endl;
			exit(1);
		}
		queue.wait_and_throw();

		free(x, queue);
		free(A, queue);
		free(xp, queue);
		/*liberação da memória após finalizaçao do programa.
		//free(Xsaved, queue);
		//free(Zsaved, queue);

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
void mult(float C[], float A[], float B[],int m,int n,int k){
	for (int i=0;i<m; i++){
		float soma=0;
		for(int j=0; j<k; j++){
			for(int incr=0; incr<k;incr++) soma+=A[i*m+incr]*B[incr+k*j];
		}
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
