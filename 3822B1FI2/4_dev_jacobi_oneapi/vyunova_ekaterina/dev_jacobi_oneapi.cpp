// Copyright 2025 Vyunova Ekaterina
#include "dev_jacobi_oneapi.h"

#include <cmath>

std::vector<float> JacobiDevONEAPI(
        const std::vector<float> a, const std::vector<float> b,
        float accuracy, sycl::device device) {
    const int n = static_cast<int>(b.size());

    sycl::queue q(device);

    float* a_dev   = sycl::malloc_device<float>(n * n, q);
    float* b_dev   = sycl::malloc_device<float>(n, q);
    float* x_cur   = sycl::malloc_device<float>(n, q);
    float* x_next  = sycl::malloc_device<float>(n, q);

    q.memcpy(a_dev, a.data(), sizeof(float) * n * n).wait();
    q.memcpy(b_dev, b.data(), sizeof(float) * n).wait();
    q.memset(x_cur,  0, sizeof(float) * n).wait();
    q.memset(x_next, 0, sizeof(float) * n).wait();

    std::vector<float> host_cur(n, 0.0f);
    std::vector<float> host_next(n, 0.0f);

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                int i = static_cast<int>(idx[0]);
                float sum = 0.0f;
                for (int j = 0; j < n; ++j) {
                    if (j != i) {
                        sum += a_dev[i * n + j] * x_cur[j];
                    }
                }
                x_next[i] = (b_dev[i] - sum) / a_dev[i * n + i];
            });
        }).wait();

        q.memcpy(host_next.data(), x_next, sizeof(float) * n).wait();
        q.memcpy(host_cur.data(),  x_cur,  sizeof(float) * n).wait();

        bool converged = true;
        for (int i = 0; i < n; ++i) {
            if (std::fabs(host_next[i] - host_cur[i]) >= accuracy) {
                converged = false;
                break;
            }
        }

        q.memcpy(x_cur, x_next, sizeof(float) * n).wait();

        if (converged) break;
    }

    sycl::free(a_dev,  q);
    sycl::free(b_dev,  q);
    sycl::free(x_cur,  q);
    sycl::free(x_next, q);

    return host_next;
}
