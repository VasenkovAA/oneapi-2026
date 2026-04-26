#include "acc_jacobi_oneapi.h"
#include <algorithm>
#include <cmath>

std::vector<float> JacobiAccONEAPI(
        const std::vector<float>& a, const std::vector<float>& b,
        float accuracy, sycl::device device) {

    std::size_t N = static_cast<std::size_t>(b.size());
    if (N == 0 || a.size() != N * N) {
        return {};
    }

    sycl::queue q(device);
    sycl::buffer<float> buf_a(a.data(), sycl::range<1>(a.size()));
    sycl::buffer<float> buf_b(b.data(), sycl::range<1>(N));

    std::vector<float> x_old(N, 0.0f);
    std::vector<float> x_new(N, 0.0f);
    sycl::buffer<float> buf_x_old(x_old.data(), sycl::range<1>(N));
    sycl::buffer<float> buf_x_new(x_new.data(), sycl::range<1>(N));

    bool use_first_as_old = true;

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        float max_diff_host = 0.0f;
        sycl::buffer<float> diff_buf(&max_diff_host, sycl::range<1>(1));
        sycl::buffer<float>& old_buf = use_first_as_old ? buf_x_old : buf_x_new;
        sycl::buffer<float>& new_buf = use_first_as_old ? buf_x_new : buf_x_old;

        q.submit([&](sycl::handler& h) {
            auto A = buf_a.get_access<sycl::access::mode::read>(h);
            auto B = buf_b.get_access<sycl::access::mode::read>(h);
            auto old_x = old_buf.get_access<sycl::access::mode::read>(h);
            auto new_x = new_buf.get_access<sycl::access::mode::write>(h);

            auto max_reduction = sycl::reduction(
                diff_buf, h, sycl::maximum<float>());

            h.parallel_for(sycl::range<1>(N), max_reduction,
                [=](sycl::id<1> i, auto& max_val) {
                    std::size_t row = i[0];
                    float sum = 0.0f;
                    for (std::size_t col = 0; col < N; ++col) {
                        if (col != row)
                            sum += A[row * N + col] * old_x[col];
                    }
                    float x_next = (B[row] - sum) / A[row * N + row];
                    new_x[row] = x_next;
                    float diff = sycl::fabs(x_next - old_x[row]);
                    max_val.combine(diff);
                });
        }).wait();
        {
            auto acc = diff_buf.get_host_access();
            max_diff_host = acc[0];
        }

        use_first_as_old = !use_first_as_old;

        if (max_diff_host < accuracy)
            break;
    }

    std::vector<float> result(N);
    {
        sycl::buffer<float>& res_buf = use_first_as_old ? buf_x_new : buf_x_old;
        auto acc = res_buf.get_host_access();
        std::copy(acc.begin(), acc.end(), result.begin());
    }
    return result;
}