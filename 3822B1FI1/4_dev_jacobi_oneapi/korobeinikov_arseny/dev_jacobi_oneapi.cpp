#include "dev_jacobi_oneapi.h"

#include <cmath>
#include <vector>

std::vector<float> JacobiDevONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {
    const std::size_t n = b.size();

    sycl::queue queue(device, sycl::property::queue::in_order{});

    std::vector<float> inv_diag(n);
    for (std::size_t i = 0; i < n; ++i) {
        inv_diag[i] = 1.0f / a[i * n + i];
    }

    float* a_dev = sycl::malloc_device<float>(a.size(), queue);
    float* b_dev = sycl::malloc_device<float>(n, queue);
    float* inv_dev = sycl::malloc_device<float>(n, queue);
    float* x_dev = sycl::malloc_device<float>(n, queue);
    float* x_new_dev = sycl::malloc_device<float>(n, queue);
    float* max_diff_dev = sycl::malloc_shared<float>(1, queue);

    queue.memcpy(a_dev, a.data(), sizeof(float) * a.size());
    queue.memcpy(b_dev, b.data(), sizeof(float) * n);
    queue.memcpy(inv_dev, inv_diag.data(), sizeof(float) * n);
    queue.fill(x_dev, 0.0f, n);

    const std::size_t local_size = 128;
    const std::size_t global_size =
        ((n + local_size - 1) / local_size) * local_size;

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        queue.submit([&](sycl::handler& handler) {
            handler.parallel_for(sycl::nd_range<1>(global_size, local_size),
                                 [=](sycl::nd_item<1> item) {
                                     const std::size_t i = item.get_global_id(0);
                                     if (i >= n) {
                                         return;
                                     }

                                     const std::size_t row = i * n;
                                     float sum = 0.0f;

                                     for (std::size_t j = 0; j < n; ++j) {
                                         if (j != i) {
                                             sum += a_dev[row + j] * x_dev[j];
                                         }
                                     }

                                     x_new_dev[i] = inv_dev[i] * (b_dev[i] - sum);
                                 });
        });

        max_diff_dev[0] = 0.0f;

        queue.submit([&](sycl::handler& handler) {
            auto reduction = sycl::reduction(max_diff_dev, sycl::maximum<float>());

            handler.parallel_for(sycl::nd_range<1>(global_size, local_size), reduction,
                                 [=](sycl::nd_item<1> item, auto& reducer) {
                                     const std::size_t i = item.get_global_id(0);
                                     if (i >= n) {
                                         return;
                                     }

                                     const float diff = sycl::fabs(x_new_dev[i] - x_dev[i]);
                                     reducer.combine(diff);
                                 });
        }).wait();

        std::swap(x_dev, x_new_dev);

        if (max_diff_dev[0] < accuracy) {
            break;
        }
    }

    std::vector<float> result(n);
    queue.memcpy(result.data(), x_dev, sizeof(float) * n).wait();

    sycl::free(a_dev, queue);
    sycl::free(b_dev, queue);
    sycl::free(inv_dev, queue);
    sycl::free(x_dev, queue);
    sycl::free(x_new_dev, queue);
    sycl::free(max_diff_dev, queue);

    return result;
}
