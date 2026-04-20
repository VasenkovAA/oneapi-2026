#include "integral_oneapi.h"
#include <algorithm>

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    if (count <= 0) return 0.0f;
    if (start > end) std::swap(start, end);

    const float d = (end - start) / static_cast<float>(count);

    sycl::queue q(device);

    float sum = 0.0f;
    {
        sycl::buffer<float, 1> sum_buf(&sum, sycl::range<1>(1));
        q.submit([&](sycl::handler& h) {
            auto red = sycl::reduction(sum_buf, h, sycl::plus<float>());
            h.parallel_for(
                sycl::range<2>(static_cast<size_t>(count), static_cast<size_t>(count)),
                red,
                [=](sycl::id<2> idx, auto& acc) {
                    const float x = start + (static_cast<float>(idx[0]) + 0.5f) * d;
                    const float y = start + (static_cast<float>(idx[1]) + 0.5f) * d;
                    acc += sycl::sin(x) * sycl::cos(y);
                }
            );
        });
    }

    return sum * d * d;
}
