#include "dev_jacobi_oneapi.h"

#include <cmath>

std::vector<float> JacobiDevONEAPI(
    const std::vector<float>& a,
    const std::vector<float>& b,
    float accuracy,
    sycl::device device
) {
    const int n = b.size();

    sycl::queue q(device);

    float* a_dev = sycl::malloc_device<float>(n * n, q);
    float* b_dev = sycl::malloc_device<float>(n, q);
    float* x_dev = sycl::malloc_device<float>(n, q);
    float* x_new_dev = sycl::malloc_device<float>(n, q);
    float* diff_dev = sycl::malloc_device<float>(1, q);

    q.memcpy(a_dev, a.data(), sizeof(float) * n * n);
    q.memcpy(b_dev, b.data(), sizeof(float) * n);

    q.memset(x_dev, 0, sizeof(float) * n);
    q.memset(x_new_dev, 0, sizeof(float) * n);

    std::vector<float> result(n);

    for (int iter = 0; iter < ITERATIONS; iter++) {
        float zero = 0.0f;
        q.memcpy(diff_dev, &zero, sizeof(float)).wait();

        q.submit([&](sycl::handler& h) {
            auto reduction = sycl::reduction(
                diff_dev, h, 0.0f, std::plus<>()
            );

            h.parallel_for(
                sycl::range<1>(n),
                reduction,
                [=](sycl::id<1> id, auto& sum_diff) {
                    int i = id[0];

                    float s = 0.0f;

                    for (int j = 0; j < n; j++) {
                        if (j != i) {
                            s += a_dev[i * n + j] * x_dev[j];
                        }
                    }

                    float new_val =
                        (b_dev[i] - s) / a_dev[i * n + i];

                    x_new_dev[i] = new_val;

                    float d = new_val - x_dev[i];
                    sum_diff += d * d;
                }
            );
            });

        q.wait();

        float diff_host;
        q.memcpy(&diff_host, diff_dev, sizeof(float)).wait();

        if (std::sqrt(diff_host) < accuracy) {
            break;
        }

        std::swap(x_dev, x_new_dev);
    }

    q.memcpy(result.data(), x_dev, sizeof(float) * n).wait();

    sycl::free(a_dev, q);
    sycl::free(b_dev, q);
    sycl::free(x_dev, q);
    sycl::free(x_new_dev, q);
    sycl::free(diff_dev, q);

    return result;
}