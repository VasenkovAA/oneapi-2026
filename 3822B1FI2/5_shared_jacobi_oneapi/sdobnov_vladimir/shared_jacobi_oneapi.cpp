#include "shared_jacobi_oneapi.h"

#include <cmath>

std::vector<float> JacobiSharedONEAPI(
    const std::vector<float>& a,
    const std::vector<float>& b,
    float accuracy,
    sycl::device device
) {
    const int n = b.size();

    sycl::queue q(device);

    float* a_mem = sycl::malloc_shared<float>(n * n, q);
    float* b_mem = sycl::malloc_shared<float>(n, q);
    float* x = sycl::malloc_shared<float>(n, q);
    float* x_new = sycl::malloc_shared<float>(n, q);
    float* diff_arr = sycl::malloc_shared<float>(n, q);

    for (int i = 0; i < n * n; i++) a_mem[i] = a[i];
    for (int i = 0; i < n; i++) b_mem[i] = b[i];

    for (int i = 0; i < n; i++) {
        x[i] = 0.0f;
        x_new[i] = 0.0f;
    }

    std::vector<float> result(n);

    for (int iter = 0; iter < ITERATIONS; iter++) {

        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            int i = id[0];

            float s = 0.0f;

            for (int j = 0; j < n; j++) {
                if (j != i) {
                    s += a_mem[i * n + j] * x[j];
                }
            }

            float new_val =
                (b_mem[i] - s) / a_mem[i * n + i];

            x_new[i] = new_val;

            float d = new_val - x[i];
            diff_arr[i] = d * d;
            });

        q.wait();

        float diff = 0.0f;
        for (int i = 0; i < n; i++) {
            diff += diff_arr[i];
        }

        if (std::sqrt(diff) < accuracy) {
            break;
        }

        std::swap(x, x_new);
    }

    for (int i = 0; i < n; i++) {
        result[i] = x[i];
    }

    sycl::free(a_mem, q);
    sycl::free(b_mem, q);
    sycl::free(x, q);
    sycl::free(x_new, q);
    sycl::free(diff_arr, q);

    return result;
}