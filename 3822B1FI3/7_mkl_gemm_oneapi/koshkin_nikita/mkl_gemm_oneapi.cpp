#include "mkl_gemm_oneapi.h"

std::vector<float> GemmMklONEAPI(const std::vector<float>& a,
                                 const std::vector<float>& b, size_t size,
                                 sycl::device device) {
  sycl::queue queue(device);

  std::vector<float> result(size * size, 0.0f);

  {
    sycl::buffer<float> a_buf(a.data(), sycl::range<1>(size * size));
    sycl::buffer<float> b_buf(b.data(), sycl::range<1>(size * size));
    sycl::buffer<float> c_buf(result.data(), sycl::range<1>(size * size));

    using oneapi::mkl::blas::row_major::gemm;
    using oneapi::mkl::transpose;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    gemm(queue, transpose::nontrans, transpose::nontrans, size, size, size,
         alpha, a_buf, size, b_buf, size, beta, c_buf, size);
  }

  queue.wait();

  return result;
}