#include "mkl_gemm_oneapi.h"
#include <oneapi/mkl.hpp>

std::vector<float> GemmMklONEAPI(
    const std::vector<float>& mat_a, const std::vector<float>& mat_b,
    size_t dim, sycl::device dev) {
    
    sycl::queue q(dev);
    std::vector<float> mat_c(dim * dim);

    float* d_a = sycl::malloc_device<float>(dim * dim, q);
    float* d_b = sycl::malloc_device<float>(dim * dim, q);
    float* d_c = sycl::malloc_device<float>(dim * dim, q);

    if (!d_a || !d_b || !d_c) return {};

    auto event_a = q.memcpy(d_a, mat_a.data(), sizeof(float) * dim * dim);
    auto event_b = q.memcpy(d_b, mat_b.data(), sizeof(float) * dim * dim);

    const float scale_a = 1.0f;
    const float scale_c = 0.0f;
    
    sycl::event gemm_done;
    try {
        gemm_done = oneapi::mkl::blas::row_major::gemm(
            q, 
            oneapi::mkl::transpose::nontrans,
            oneapi::mkl::transpose::nontrans,
            dim, dim, dim,                   
            scale_a,                          
            d_a, dim,                         
            d_b, dim,                        
            scale_c,                       
            d_c, dim,                         
            {event_a, event_b}
        );
    } catch (sycl::exception const& e) {
        sycl::free(d_a, q); sycl::free(d_b, q); sycl::free(d_c, q);
        return {};
    }

    q.memcpy(mat_c.data(), d_c, sizeof(float) * dim * dim, gemm_done).wait();

    sycl::free(d_a, q);
    sycl::free(d_b, q);
    sycl::free(d_c, q);

    return mat_c;
}