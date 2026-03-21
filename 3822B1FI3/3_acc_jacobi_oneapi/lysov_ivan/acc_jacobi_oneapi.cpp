#include "acc_jacobi_oneapi.h"

#include <utility>

std::vector<float> JacobiAccONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {
    const size_t n = b.size();

    if (n == 0) {
        return {};
    }

    if (a.size() != n * n) {
        return {};
    }

    if (accuracy < 0.0f) {
        accuracy = 0.0f;
    }

    sycl::queue q(device);

    std::vector<float> x_old(n, 0.0f);
    std::vector<float> x_new(n, 0.0f);

    {
        sycl::buffer<float, 1> a_buffer(a.data(), sycl::range<1>(a.size()));
        sycl::buffer<float, 1> b_buffer(b.data(), sycl::range<1>(b.size()));
        sycl::buffer<float, 1> x_old_buffer(x_old.data(), sycl::range<1>(n));
        sycl::buffer<float, 1> x_new_buffer(x_new.data(), sycl::range<1>(n));

        for (int iter = 0; iter < ITERATIONS; ++iter) {
            float max_diff = 0.0f;

            {
                sycl::buffer<float, 1> diff_buffer(&max_diff, sycl::range<1>(1));

                q.submit([&](sycl::handler& h) {
                    auto a_acc = a_buffer.get_access<sycl::access::mode::read>(h);
                    auto b_acc = b_buffer.get_access<sycl::access::mode::read>(h);
                    auto x_old_acc = x_old_buffer.get_access<sycl::access::mode::read>(h);
                    auto x_new_acc = x_new_buffer.get_access<sycl::access::mode::write>(h);

                    auto max_reduction =
                        sycl::reduction(diff_buffer, h, sycl::maximum<float>());

                    h.parallel_for(
                        sycl::range<1>(n),
                        max_reduction,
                        [=](sycl::id<1> idx, auto& max_val) {
                            const size_t i = idx[0];
                            const size_t row_offset = i * n;

                            float row_sum = 0.0f;
                            for (size_t j = 0; j < n; ++j) {
                                if (j != i) {
                                    row_sum += a_acc[row_offset + j] * x_old_acc[j];
                                }
                            }

                            const float diag = a_acc[row_offset + i];
                            const float new_value = (b_acc[i] - row_sum) / diag;

                            x_new_acc[i] = new_value;

                            const float diff = sycl::fabs(new_value - x_old_acc[i]);
                            max_val.combine(diff);
                        });
                }).wait();
            }

            if (max_diff < accuracy) {
                break;
            }

            q.submit([&](sycl::handler& h) {
                auto src = x_new_buffer.get_access<sycl::access::mode::read>(h);
                auto dst = x_old_buffer.get_access<sycl::access::mode::write>(h);

                h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
                    dst[idx] = src[idx];
                });
            }).wait();
        }
    }

    return x_new;
}