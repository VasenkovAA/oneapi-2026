#include "acc_jacobi_oneapi.h"
#include <algorithm>
#include <cmath>

using buftype = sycl::buffer<float>;

std::vector<float> JacobiAccONEAPI(const std::vector<float> &a,
                                   const std::vector<float> &b, float accuracy,
                                   sycl::device device) {
  size_t size = b.size();
  std::vector<float> prev_res(size, 0.0f);
  std::vector<float> res(size, 0.0f);

  sycl::queue gpu_queue(device);

  buftype buf_a(a.data(), a.size());
  buftype buf_b(b.data(), b.size());
  buftype buf_prev_res(prev_res.data(), prev_res.size());
  buftype buf_res(res.data(), res.size());

  for (int epoch = 0; epoch < ITERATIONS; ++epoch) {
    gpu_queue
        .submit([&](sycl::handler &cgh) {
          auto in_a = buf_a.get_access<sycl::access::mode::read>(cgh);
          auto in_b = buf_b.get_access<sycl::access::mode::read>(cgh);
          auto in_prev_res =
              buf_prev_res.get_access<sycl::access::mode::read>(cgh);
          auto in_res = buf_res.get_access<sycl::access::mode::write>(cgh);

          cgh.parallel_for(sycl::range<1>(size), [=](sycl::id<1> id) {
            int idx = id.get(0);
            float next_res = 0;

            for (int indx = 0; indx < size; indx++) {
              if (idx == indx) {
                next_res += in_b[indx];
              } else {
                next_res -= in_a[idx * size + indx] * in_prev_res[indx];
              }
            }
            next_res /= in_a[idx * size + idx];
            in_res[idx] = next_res;
          });
        })
        .wait();

    auto buf_prev_res_host = buf_prev_res.get_host_access(sycl::read_write);
    auto buf_res_host = buf_res.get_host_access(sycl::read_only);

    float norm = 0.0f;
    for (int idx = 0; idx < size; ++idx) {
      float el = std::fabs(buf_res_host[idx] - buf_prev_res_host[idx]);
      if (el > norm)
        norm = el;
      buf_prev_res_host[idx] = buf_res_host[idx];
    }

    if (norm < accuracy) {
      break;
    }
  }

  auto host_final = buf_prev_res.get_host_access(sycl::read_only);
  std::vector<float> final(size);
  for (int i = 0; i < size; ++i) {
    final[i] = host_final[i];
  }

  return final;
}
