#include "acc_jacobi_oneapi.h"
#include <cmath>
#include <utility>

std::vector<float> JacobiAccONEAPI(
    const std::vector<float>& a,
    const std::vector<float>& b,
    float accuracy,
    sycl::device device) {

    sycl::queue queue{ device, sycl::property::queue::in_order {} };
    size_t n = b.size();
    std::vector<float> x_curr(n, 0.0f);
    std::vector<float> x_next(n, 0.0f);

    sycl::buffer<float, 1> a_buf{ a.data(), sycl::range<1>(a.size()) };
    sycl::buffer<float, 1> b_buf{ b.data(), sycl::range<1>(n) };
    sycl::buffer<float, 1> x_curr_buf{ x_curr.data(), sycl::range<1>(n) };
    sycl::buffer<float, 1> x_next_buf{ x_next.data(), sycl::range<1>(n) };

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        float it = 0.0f;
        sycl::buffer<float, 1> diff_buf{ &it, 1};
        queue.submit([&](sycl::handler& cgh) {
            auto a_acc = a_buf.get_access<sycl::access::mode::read>(cgh);
            auto b_acc = b_buf.get_access<sycl::access::mode::read>(cgh);
            auto x_c_acc = x_curr_buf.get_access<sycl::access::mode::read>(cgh);
            auto x_n_acc = x_next_buf.get_access<sycl::access::mode::write>(cgh);

            auto reduct = sycl::reduction(diff_buf, cgh, sycl::plus<float>());

            cgh.parallel_for(sycl::range<1>(n), [=](sycl::item<1> item) {
                size_t i = item[0];
                float sum = 0.0f;
                size_t size = i * n;
                for (size_t j = 0; j < n; ++j) {
                    if (j != i) {
                        sum += a_acc[i * n + j] * x_c_acc[j];
                    }
                }
                x_n_acc[i] = (b_acc[i] - sum) / a_acc[i * n + i];
                });
            });
       
            auto diff_host = diff_buf.get_host_access();
            if (diff_host[0] < accuracy) {
               break;
            }
            std::swap(x_curr_buf, x_next_buf);
    }

         std::vector<float> result(n);
         auto last = x_curr_buf.get_host_access();
         for (size_t i = 0; i < n; ++i) {
             result[i] = last[i];
         }

    return result;
}