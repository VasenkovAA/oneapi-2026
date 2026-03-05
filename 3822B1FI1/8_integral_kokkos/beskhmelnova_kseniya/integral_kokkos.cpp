#include "integral_kokkos.h"
#include <cmath>

float IntegralKokkos(float start, float end, int count) {
    using ExecSpace = Kokkos::SYCL;
    
    const float step = (end - start) / static_cast<float>(count);
    const float area = step * step;

    double sum_sin = 0.0;
    Kokkos::parallel_reduce(
        "IntegralSin",
        Kokkos::RangePolicy<ExecSpace>(0, count),
        KOKKOS_LAMBDA(const int i, double& lsum) {
            const float x = start + step * (static_cast<float>(i) + 0.5f);
            lsum += Kokkos::sin(x);
        },
        sum_sin
    );

    double sum_cos = 0.0;
    Kokkos::parallel_reduce(
        "IntegralCos",
        Kokkos::RangePolicy<ExecSpace>(0, count),
        KOKKOS_LAMBDA(const int j, double& lsum) {
            const float y = start + step * (static_cast<float>(j) + 0.5f);
            lsum += Kokkos::cos(y);
        },
        sum_cos
    );

    return static_cast<float>(sum_sin * sum_cos * area);
}
