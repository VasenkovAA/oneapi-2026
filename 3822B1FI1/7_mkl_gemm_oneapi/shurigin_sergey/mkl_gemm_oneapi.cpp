#include "mkl_gemm_oneapi.h"
#include <cstdint>
#include <oneapi/mkl.hpp>

std::vector<float> GemmMklONEAPI(
        const std::vector<float>& a,
        const std::vector<float>& b,
        size_t size,
        sycl::device device) {
    
    sycl::queue q(device, sycl::property::queue::in_order{});

    const size_t total = size * size;
    const std::int64_t n = static_cast<std::int64_t>(size);
    
    std::vector<float> result(total);

    float* d_a = sycl::aligned_alloc_device<float>(64, total, q);
    float* d_b = sycl::aligned_alloc_device<float>(64, total, q);
    float* d_c = sycl::aligned_alloc_device<float>(64, total, q);

    q.memcpy(d_a, a.data(), total * sizeof(float));
    q.memcpy(d_b, b.data(), total * sizeof(float));

    oneapi::mkl::blas::row_major::gemm(
            q,
            oneapi::mkl::transpose::nontrans,
            oneapi::mkl::transpose::nontrans,
            n, n, n,
            1.0f, // alpha
            d_a, n, // lda = n
            d_b, n, // ldb = n
            0.0f, // beta
            d_c, n // ldc = n
    );

    q.memcpy(result.data(), d_c, total * sizeof(float)).wait();

    sycl::free(d_a, q);
    sycl::free(d_b, q);
    sycl::free(d_c, q);

    return result;
}