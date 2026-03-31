#include "shared_jacobi_oneapi.h"
#include <cmath>

std::vector<float> JacobiSharedONEAPI(
    const std::vector<float>& a, const std::vector<float>& b,
    float accuracy, sycl::device device) {

    size_t n = b.size();

    if (n == 0 || a.size() != n * n) {
        return std::vector<float>();
    }

    sycl::queue queue(device);

    float* A = sycl::malloc_shared<float>(a.size(), queue);
    float* B = sycl::malloc_shared<float>(b.size(), queue);
    float* X = sycl::malloc_shared<float>(n, queue);
    float* X_new = sycl::malloc_shared<float>(n, queue);

    for (size_t i = 0; i < a.size(); i++) {
        A[i] = a[i];
    }

    for (size_t i = 0; i < n; i++) {
        B[i] = b[i];
        X[i] = 0.0f;
    }

    bool converged = false;
    int iteration = 0;

    while (!converged && iteration < ITERATIONS) {
        queue.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            size_t row = i[0];
            float sum = 0.0f;
            float diag = A[row * n + row];

            for (size_t j = 0; j < n; j++) {
                if (j != row) {
                    sum += A[row * n + j] * X[j];
                }
            }

            X_new[row] = (B[row] - sum) / diag;
            });

        queue.wait();

        converged = true;

        for (size_t i = 0; i < n; i++) {
            float diff = std::fabs(X_new[i] - X[i]);

            if (diff >= accuracy) {
                converged = false;
            }

            X[i] = X_new[i];
        }

        iteration++;
    }

    std::vector<float> result(n);
    for (size_t i = 0; i < n; i++) {
        result[i] = X[i];
    }

    sycl::free(A, queue);
    sycl::free(B, queue);
    sycl::free(X, queue);
    sycl::free(X_new, queue);

    return result;
}
