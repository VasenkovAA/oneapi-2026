#include "dev_jacobi_oneapi.h"

#include <cmath>

std::vector<float> JacobiDevONEAPI(
  const std::vector<float>& a,
  const std::vector<float>& b,
  float accuracy,
  sycl::device device) {

  const int n = b.size();

  std::vector<float> curr_host(n, 0.0f);
  std::vector<float> prev_host(n, 0.0f);

  sycl::queue q(device);

  float* a_dev = sycl::malloc_device<float>(a.size(), q);
  float* b_dev = sycl::malloc_device<float>(b.size(), q);
  float* prev_dev = sycl::malloc_device<float>(n, q);
  float* curr_dev = sycl::malloc_device<float>(n, q);

  q.memcpy(a_dev, a.data(), sizeof(float) * a.size()).wait();
  q.memcpy(b_dev, b.data(), sizeof(float) * b.size()).wait();
  q.memset(prev_dev, 0, sizeof(float) * n).wait();
  q.memset(curr_dev, 0, sizeof(float) * n).wait();

  for (int iter = 0; iter < ITERATIONS; ++iter) {
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
      int i = id[0];
      float value = b_dev[i];

      for (int j = 0; j < n; ++j) {
        if (i != j) {
          value -= a_dev[i * n + j] * prev_dev[j];
        }
      }

      curr_dev[i] = value / a_dev[i * n + i];
      }).wait();

    q.memcpy(curr_host.data(), curr_dev, sizeof(float) * n).wait();
    q.memcpy(prev_host.data(), prev_dev, sizeof(float) * n).wait();

    bool ok = true;
    for (int i = 0; i < n; ++i) {
      if (std::fabs(curr_host[i] - prev_host[i]) >= accuracy) {
        ok = false;
      }
    }

    q.memcpy(prev_dev, curr_dev, sizeof(float) * n).wait();

    if (ok) {
      break;
    }
  }

  sycl::free(a_dev, q);
  sycl::free(b_dev, q);
  sycl::free(prev_dev, q);
  sycl::free(curr_dev, q);

  return curr_host;
}