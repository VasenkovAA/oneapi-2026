#include "dev_jacobi_oneapi.h"

#include <algorithm>
#include <cmath>
#include <vector>

std::vector<float> JacobiDevONEAPI(
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

        float* a_dev = sycl::malloc_device<float>(a.size(), q);
        float* b_dev = sycl::malloc_device<float>(b.size(), q);
        float* x0_dev = sycl::malloc_device<float>(n, q);
        float* x1_dev = sycl::malloc_device<float>(n, q);
        float* max_diff_dev = sycl::malloc_device<float>(1, q);

        if (a_dev == nullptr || b_dev == nullptr ||
            x0_dev == nullptr || x1_dev == nullptr ||
            max_diff_dev == nullptr) {
            if (a_dev) sycl::free(a_dev, q);
            if (b_dev) sycl::free(b_dev, q);
            if (x0_dev) sycl::free(x0_dev, q);
            if (x1_dev) sycl::free(x1_dev, q);
            if (max_diff_dev) sycl::free(max_diff_dev, q);
            return {};
        }

        q.memcpy(a_dev, a.data(), sizeof(float) * a.size());
        q.memcpy(b_dev, b.data(), sizeof(float) * b.size());
        q.fill(x0_dev, 0.0f, n);
        q.fill(x1_dev, 0.0f, n);
        q.wait();

        float* x_current = x0_dev;
        float* x_next = x1_dev;

        float max_diff_host = 0.0f;

        for (int iter = 0; iter < ITERATIONS; ++iter) {
            q.fill(max_diff_dev, 0.0f, 1).wait();

            q.submit([&](sycl::handler& h) {
                auto max_red = sycl::reduction(
                    max_diff_dev, 0.0f, sycl::maximum<float>());

                h.parallel_for(
                    sycl::range<1>(n),
                    max_red,
                    [=](sycl::id<1> idx, auto& max_val) {
                        const std::size_t i = idx[0];
                        const std::size_t row = i * n;

                        float sum = 0.0f;
                        for (std::size_t j = 0; j < n; ++j) {
                            if (j != i) {
                                sum += a_dev[row + j] * x_current[j];
                            }
                        }

                        const float new_value =
                            (b_dev[i] - sum) / a_dev[row + i];

                        x_next[i] = new_value;

                        const float diff = sycl::fabs(new_value - x_current[i]);
                        max_val.combine(diff);
                    });
            });

            q.memcpy(&max_diff_host, max_diff_dev, sizeof(float)).wait();

            std::swap(x_current, x_next);

            if (max_diff_host < accuracy) {
                break;
            }
        }

        std::vector<float> result(n);
        q.memcpy(result.data(), x_current, sizeof(float) * n).wait();

        sycl::free(a_dev, q);
        sycl::free(b_dev, q);
        sycl::free(x0_dev, q);
        sycl::free(x1_dev, q);
        sycl::free(max_diff_dev, q);

        return result;
    } catch (const sycl::exception&) {
        return {};
    }
}