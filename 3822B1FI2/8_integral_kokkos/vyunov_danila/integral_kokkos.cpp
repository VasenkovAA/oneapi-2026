#include "integral_kokkos.h"

#include <cmath>

float IntegralKokkos(float start, float end, int count) {
    const float step = (end - start) / static_cast<float>(count);
    float result = 0.0f;

    Kokkos::parallel_reduce(
        "IntegralKokkos",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {count, count}),
        KOKKOS_LAMBDA(int i, int j, float& sum) {
            float x = start + step * (static_cast<float>(i) + 0.5f);
            float y = start + step * (static_cast<float>(j) + 0.5f);
            sum += Kokkos::sin(x) * Kokkos::cos(y);
        },
        result
    );

    return result * step * step;
}
