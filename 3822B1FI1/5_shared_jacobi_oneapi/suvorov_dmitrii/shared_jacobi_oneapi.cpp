#include "shared_jacobi_oneapi.h"

#include <cmath>
#include <vector>

std::vector<float> JacobiSharedONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {
    if (accuracy <= 0.0f) {
        accuracy = 1e-6f;
    }

    const std::size_t n = b.size();
    if (n == 0 || a.size() != n * n) {
        return {};
    }

    try {
        sycl::queue q(device);

        float* a_sh = sycl::malloc_shared<float>(a.size(), q);
        float* b_sh = sycl::malloc_shared<float>(b.size(), q);
        float* x0_sh = sycl::malloc_shared<float>(n, q);
        float* x1_sh = sycl::malloc_shared<float>(n, q);
        float* max_diff_sh = sycl::malloc_shared<float>(1, q);

        if (a_sh == nullptr || b_sh == nullptr ||
            x0_sh == nullptr || x1_sh == nullptr ||
            max_diff_sh == nullptr) {
            if (a_sh) sycl::free(a_sh, q);
            if (b_sh) sycl::free(b_sh, q);
            if (x0_sh) sycl::free(x0_sh, q);
            if (x1_sh) sycl::free(x1_sh, q);
            if (max_diff_sh) sycl::free(max_diff_sh, q);
            return {};
        }

        for (std::size_t i = 0; i < a.size(); ++i) {
            a_sh[i] = a[i];
        }
        for (std::size_t i = 0; i < n; ++i) {
            b_sh[i] = b[i];
            x0_sh[i] = 0.0f;
            x1_sh[i] = 0.0f;
        }

        float* x_current = x0_sh;
        float* x_next = x1_sh;

        for (int iter = 0; iter < ITERATIONS; ++iter) {
            *max_diff_sh = 0.0f;

            q.submit([&](sycl::handler& h) {
                auto max_red = sycl::reduction(
                    max_diff_sh, 0.0f, sycl::maximum<float>());

                h.parallel_for(
                    sycl::range<1>(n),
                    max_red,
                    [=](sycl::id<1> idx, auto& max_val) {
                        const std::size_t i = idx[0];
                        const std::size_t row = i * n;

                        float sum = 0.0f;
                        for (std::size_t j = 0; j < n; ++j) {
                            if (j != i) {
                                sum += a_sh[row + j] * x_current[j];
                            }
                        }

                        const float new_value =
                            (b_sh[i] - sum) / a_sh[row + i];

                        x_next[i] = new_value;

                        const float diff =
                            sycl::fabs(new_value - x_current[i]);
                        max_val.combine(diff);
                    });
            });

            q.wait();

            std::swap(x_current, x_next);

            if (*max_diff_sh < accuracy) {
                break;
            }
        }

        std::vector<float> result(n);
        for (std::size_t i = 0; i < n; ++i) {
            result[i] = x_current[i];
        }

        sycl::free(a_sh, q);
        sycl::free(b_sh, q);
        sycl::free(x0_sh, q);
        sycl::free(x1_sh, q);
        sycl::free(max_diff_sh, q);

        return result;
    } catch (const sycl::exception&) {
        return {};
    }
}