#include "dev_jacobi_oneapi.h"

std::vector<float> JacobiDevONEAPI(const std::vector<float>& a,
                                   const std::vector<float>& b, float accuracy,
                                   sycl::device device) {
  sycl::queue queue(device);

  const size_t n = b.size();

  float* A = sycl::malloc_device<float>(n * n, queue);
  float* B = sycl::malloc_device<float>(n, queue);
  float* X_old = sycl::malloc_device<float>(n, queue);
  float* X_new = sycl::malloc_device<float>(n, queue);
  float* INV = sycl::malloc_device<float>(n, queue);
  float* error = sycl::malloc_shared<float>(1, queue);

  queue.memcpy(A, a.data(), sizeof(float) * n * n);
  queue.memcpy(B, b.data(), sizeof(float) * n);

  std::vector<float> inv_diag(n);
  std::vector<float> x_init(n, 0.0f);

  for (size_t i = 0; i < n; ++i) {
    inv_diag[i] = 1.0f / a[i * n + i];
  }

  queue.memcpy(INV, inv_diag.data(), sizeof(float) * n);
  queue.memcpy(X_old, x_init.data(), sizeof(float) * n);

  const size_t local_size = 256;
  const size_t global_size = ((n + local_size - 1) / local_size) * local_size;

  for (int iter = 0; iter < ITERATIONS; ++iter) {
    *error = 0.0f;

    queue.parallel_for(
        sycl::nd_range<1>(global_size, local_size), [=](sycl::nd_item<1> item) {
          size_t i = item.get_global_id(0);
          if (i >= n) return;

          float sigma = 0.0f;

#pragma unroll 4
          for (size_t j = 0; j < n; ++j) {
            if (j != i) {
              sigma += A[i * n + j] * X_old[j];
            }
          }

          float new_val = (B[i] - sigma) * INV[i];
          X_new[i] = new_val;

          float diff = sycl::fabs(new_val - X_old[i]);

          sycl::atomic_ref<float, sycl::memory_order::relaxed,
                           sycl::memory_scope::device,
                           sycl::access::address_space::global_space>
              atomic_error(*error);

          atomic_error.fetch_add(diff);
        });

    queue.wait();

    if (*error < accuracy) break;

    std::swap(X_old, X_new);
  }

  std::vector<float> result(n);
  queue.memcpy(result.data(), X_old, sizeof(float) * n).wait();

  sycl::free(A, queue);
  sycl::free(B, queue);
  sycl::free(X_old, queue);
  sycl::free(X_new, queue);
  sycl::free(INV, queue);
  sycl::free(error, queue);

  return result;
}