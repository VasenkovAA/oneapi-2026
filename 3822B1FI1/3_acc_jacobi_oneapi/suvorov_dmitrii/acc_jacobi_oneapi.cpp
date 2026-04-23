#include "acc_jacobi_oneapi.h"

#include <cmath>
#include <vector>

std::vector<float> JacobiAccONEAPI(
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

        sycl::buffer<float, 1> a_buf(a.data(), sycl::range<1>(a.size()));
        sycl::buffer<float, 1> b_buf(b.data(), sycl::range<1>(b.size()));

        std::vector<float> x0(n, 0.0f);
        std::vector<float> x1(n, 0.0f);

        sycl::buffer<float, 1> x0_buf(x0.data(), sycl::range<1>(n));
        sycl::buffer<float, 1> x1_buf(x1.data(), sycl::range<1>(n));

        sycl::buffer<float, 1>* x_current = &x0_buf;
        sycl::buffer<float, 1>* x_next = &x1_buf;

        for (int iter = 0; iter < ITERATIONS; ++iter) {
            float max_diff_host = 0.0f;
            sycl::buffer<float, 1> max_diff_buf(&max_diff_host, sycl::range<1>(1));

            q.submit([&](sycl::handler& h) {
                auto a_acc = a_buf.get_access<sycl::access::mode::read>(h);
                auto b_acc = b_buf.get_access<sycl::access::mode::read>(h);
                auto x_cur_acc = x_current->get_access<sycl::access::mode::read>(h);
                auto x_next_acc = x_next->get_access<sycl::access::mode::write>(h);

                auto max_red = sycl::reduction(
                    max_diff_buf, h, 0.0f, sycl::maximum<float>());

                h.parallel_for(
                    sycl::range<1>(n),
                    max_red,
                    [=](sycl::id<1> idx, auto& max_val) {
                        const std::size_t i = idx[0];
                        const std::size_t row = i * n;

                        float sum = 0.0f;
                        for (std::size_t j = 0; j < n; ++j) {
                            if (j != i) {
                                sum += a_acc[row + j] * x_cur_acc[j];
                            }
                        }

                        const float new_value = (b_acc[i] - sum) / a_acc[row + i];
                        x_next_acc[i] = new_value;

                        const float diff = sycl::fabs(new_value - x_cur_acc[i]);
                        max_val.combine(diff);
                    });
            });

            q.wait();

            std::swap(x_current, x_next);

            if (max_diff_host < accuracy) {
                break;
            }
        }

        std::vector<float> result(n);
        {
            sycl::host_accessor result_acc(*x_current, sycl::read_only);
            for (std::size_t i = 0; i < n; ++i) {
                result[i] = result_acc[i];
            }
        }

        return result;
    } catch (const sycl::exception&) {
        return {};
    }
}