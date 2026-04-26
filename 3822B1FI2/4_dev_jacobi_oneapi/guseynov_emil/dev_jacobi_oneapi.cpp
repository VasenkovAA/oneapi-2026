#include "dev_jacobi_oneapi.h"

std::vector<float> JacobiDevONEAPI(
    const std::vector<float>& mat_a, const std::vector<float>& vec_b,
    float tolerance, sycl::device dev) {
    
    const size_t sz = vec_b.size();
    std::vector<float> result(sz);

    sycl::queue q(dev, {sycl::property::queue::in_order()});

    float* d_a    = sycl::malloc_device<float>(sz * sz, q);
    float* d_b    = sycl::malloc_device<float>(sz, q);
    float* d_curr = sycl::malloc_device<float>(sz, q);
    float* d_next = sycl::malloc_device<float>(sz, q);
    float* d_diff = sycl::malloc_device<float>(1, q);

    if (!d_a || !d_b || !d_curr || !d_next || !d_diff) {
        return {};
    }

    q.memcpy(d_a, mat_a.data(), sizeof(float) * sz * sz);
    q.memcpy(d_b, vec_b.data(), sizeof(float) * sz);
    q.fill(d_curr, 0.0f, sz);
    q.wait();

    int step = 0;
    float current_err = 0.0f;

    do {
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(sz), [=](sycl::id<1> idx) {
                size_t i = idx[0];
                float sum = 0.0f;
                for (size_t j = 0; j < sz; ++j) {
                    if (i != j) sum += d_a[i * sz + j] * d_curr[j];
                }
                d_next[i] = (d_b[i] - sum) / d_a[i * sz + i];
            });
        });

        q.fill(d_diff, 0.0f, 1);

        q.submit([&](sycl::handler& h) {
            auto red = sycl::reduction(d_diff, 0.0f, sycl::maximum<float>());
            h.parallel_for(sycl::range<1>(sz), red, [=](sycl::id<1> idx, auto& max_v) {
                max_v.combine(sycl::fabs(d_next[idx] - d_curr[idx]));
            });
        });

        q.memcpy(&current_err, d_diff, sizeof(float)).wait();

        std::swap(d_curr, d_next);
        step++;
    } while (step < ITERATIONS && current_err >= tolerance);

    q.memcpy(result.data(), d_curr, sizeof(float) * sz).wait();

    sycl::free(d_a, q);
    sycl::free(d_b, q);
    sycl::free(d_curr, q);
    sycl::free(d_next, q);
    sycl::free(d_diff, q);

    return result;
}