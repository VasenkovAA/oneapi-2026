#include "mkl_gemm_oneapi.h"
#include <oneapi/mkl.hpp>

std::vector<float> GemmMklONEAPI(const std::vector<float> &a,
                                 const std::vector<float> &b, size_t size,
                                 sycl::device device) {

  sycl::queue gpu_queue(device);
  std::vector<float> res(size * size, 0.0f);

  sycl::buffer<float> in_a(a.data(), sycl::range<1>(a.size()));
  sycl::buffer<float> in_b(b.data(), sycl::range<1>(b.size()));
  sycl::buffer<float> in_c(res.data(), sycl::range<1>(res.size()));

   oneapi::mkl::blas::row_major::gemm(gpu_queue,
                          oneapi::mkl::transpose::nontrans,
                          oneapi::mkl::transpose::nontrans, size, size, size,
                          1.0f, in_a, size, in_b, size, 0.0f, in_c, size);
  gpu_queue.wait();

  return res;
}
