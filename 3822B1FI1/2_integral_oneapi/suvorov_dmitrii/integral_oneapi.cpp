#include "integral_oneapi.h"
#include <sycl/sycl.hpp>

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    if (count <= 0) {
        return 0.0f;
    }

    const float step = (end - start) / static_cast<float>(count);

    sycl::queue q(device);

    float sum_sin = 0.0f;
    float sum_cos = 0.0f;

    {
        sycl::buffer<float, 1> sin_buf(&sum_sin, sycl::range<1>(1));
        sycl::buffer<float, 1> cos_buf(&sum_cos, sycl::range<1>(1));

        q.submit([&](sycl::handler& h) {
            auto sin_red = sycl::reduction(
                sin_buf, h, 0.0f, sycl::plus<float>());

            auto cos_red = sycl::reduction(
                cos_buf, h, 0.0f, sycl::plus<float>());

            h.parallel_for(sycl::range<1>(static_cast<std::size_t>(count)),
                           sin_red,
                           cos_red,
                           [=](sycl::id<1> idx, auto& sin_acc, auto& cos_acc) {
                               const float i = static_cast<float>(idx[0]);
                               const float mid = start + (i + 0.5f) * step;

                               sin_acc += sycl::sin(mid);
                               cos_acc += sycl::cos(mid);
                           });
        });

        q.wait();
    }

    return (sum_sin * step) * (sum_cos * step);
}