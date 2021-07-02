#include <iostream>
#include <limits>
#include <random>

#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"

double GetVolt() {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);
    double w = 0 + 4*distribution(generator);
    double z = 14.4 + w;
    return z;
}

double SimpleKalman(double z);

int main()
{
    try {
        auto nontransM = oneapi::mkl::transpose::nontrans;
        auto transM = oneapi::mkl::transpose::trans;

        double alpha = 1.0;
        double beta = 0.0;

        sycl::queue device_queue{sycl::default_selector{}};

        std::cout << "Device: "
                  << device_queue.get_device().get_info<sycl::info::device::name>()
                  << std::endl;

        double dt = 0.2, first = 0.0, volt;
        int Nsamples = 51;
        double t[Nsamples];
        
        auto Xsaved = sycl::malloc_shared<double>(Nsamples, device_queue);
        auto Zsaved = sycl::malloc_shared<double>(Nsamples, device_queue);

        auto A = sycl::malloc_shared<double>(1, device_queue); A[0] = 1.0;
        auto H = sycl::malloc_shared<double>(1, device_queue); H[0] = 1.0;
        auto Q = sycl::malloc_shared<double>(1, device_queue); Q[0] = 0.0;
        auto R = sycl::malloc_shared<double>(1, device_queue); R[0] = 4.0;
        auto x = sycl::malloc_shared<double>(1, device_queue); x[0] = 14.0;
        auto P = sycl::malloc_shared<double>(1, device_queue); P[0] = 6.0;

        auto xp = sycl::malloc_shared<double>(1, device_queue);
        auto Pp = sycl::malloc_shared<double>(1, device_queue);
        auto K  = sycl::malloc_shared<double>(1, device_queue);

        auto AP    = sycl::malloc_shared<double>(1, device_queue);
        auto PpHT  = sycl::malloc_shared<double>(1, device_queue);
        auto HpHTR = sycl::malloc_shared<double>(1, device_queue);
        
        std::int64_t scratchpad_size = oneapi::mkl::lapack::getrf_scratchpad_size<double>(device_queue, 1, 1, 1);
        double* scratchpad = sycl::malloc_shared<double>(scratchpad_size, device_queue);
        
        ipiv???

        for (int i = 0; i < Nsamples; i++) {
            t[i] = first += dt;
            Zsaved[i] = GetVolt();
            // xp = A * x
            oneapi::mkl::blas::row_major::gemm(device_queue, nontransM, nontransM, 1, 1, 1, alpha, A, 1, x, 1, beta, xp, 1);
            // Pp = A * P * A' + Q
              // AP = A * P
            oneapi::mkl::blas::row_major::gemm(device_queue, nontransM, nontransM, 1, 1, 1, alpha, A, 1, P, 1, beta, AP, 1);
              // Pp = AP * A'
            oneapi::mkl::blas::row_major::gemm(device_queue, nontransM, transM, 1, 1, 1, alpha, AP, 1, A, 1, beta, Pp, 1);
              // Pp = Pp + Q
            oneapi::mkl::blas::axpy(device_queue, 1, 1, Q, 1, Pp, 1);
            // K = Pp * H' * inv(H * Pp * H' + R)
              // PpHT = Pp * H'
            oneapi::mkl::blas::row_major::gemm(device_queue, nontransM, transM, 1, 1, 1, alpha, Pp, 1, H, 1, beta, PpHT, 1);
              // HpHTR = H * Pp * H'
            oneapi::mkl::blas::row_major::gemm(device_queue, nontransM, nontransM, 1, 1, 1, alpha, PpHT, 1, H, 1, beta, HpHTR, 1);
              // HpHTR = HpHTR + R
            oneapi::mkl::blas::axpy(device_queue, 1, 1, R, 1, HpHTR, 1);
              // HpHTR = inv(HpHTR)
            oneapi::mkl::lapack::getrf(device_queue, 1, 1, HpHTR, 1, ipiv, scratchpad, scratchpad_size);
            oneapi::mkl::lapack::getri(device_queue, 1, 1, HpHTR, 1, ipiv, scratchpad, scratchpad_size);
              // K = PpHT * HpHTR
            oneapi::mkl::blas::row_major::gemm(device_queue, nontransM, nontransM, 1, 1, 1, alpha, HpHTR, 1, PpHT, 1, beta, K, 1);
            // x = xp + K * (z - H * xp)
            // P = Pp - K * H * Pp
            // volt = x
            Xsaved[i] = volt;
        }

        if (!Xsaved || !Zsaved) {
            std::cerr << "Could not allocate memory for vectors." << std::endl;
            exit(1);
        }
        
        device_queue.wait_and_throw();

        free(Xsaved, device_queue);
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
        ...
        
    } catch (const std::exception &e) {
        std::cerr << "An exception occurred: "
                  << e.what() << std::endl;
        exit(1);
    }
}
