#include "acc_jacobi_oneapi.h"
#include <cmath>

std::vector<float> JacobiAccONEAPI(
    const std::vector<float>& a, const std::vector<float>& b,
    float accuracy, sycl::device device)
{
    size_t n = b.size();
    std::vector<float> x(n, 0.0f);
    std::vector<float> x_new(n, 0.0f);

    sycl::queue queue(device);

    for (int iter = 0; iter < ITERATIONS; ++iter)
    {
        {
            sycl::buffer<float> a_buf(a.data(), sycl::range<1>(n * n));
            sycl::buffer<float> b_buf(b.data(), sycl::range<1>(n));
            sycl::buffer<float> x_buf(x.data(), sycl::range<1>(n));
            sycl::buffer<float> x_new_buf(x_new.data(), sycl::range<1>(n));

            queue.submit([&](sycl::handler& cgh)
                {
                    auto a_acc = a_buf.get_access<sycl::access::mode::read>(cgh);
                    auto b_acc = b_buf.get_access<sycl::access::mode::read>(cgh);
                    auto x_acc = x_buf.get_access<sycl::access::mode::read>(cgh);
                    auto x_new_acc = x_new_buf.get_access<sycl::access::mode::write>(cgh);

                    cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx)
                        {
                            size_t i = idx[0];
                            float sum = b_acc[i];

                            for (size_t j = 0; j < n; ++j)
                            {
                                if (i != j)
                                {
                                    sum -= a_acc[i * n + j] * x_acc[j];
                                }
                            }

                            x_new_acc[i] = sum / a_acc[i * n + i];
                        });
                });
        }

        float max_diff = 0.0f;
        for (size_t i = 0; i < n; ++i)
        {
            float diff = std::fabs(x_new[i] - x[i]);
            if (diff > max_diff)
            {
                max_diff = diff;
            }
            x[i] = x_new[i];
        }

        if (max_diff < accuracy)
        {
            break;
        }
    }

    return x;
}