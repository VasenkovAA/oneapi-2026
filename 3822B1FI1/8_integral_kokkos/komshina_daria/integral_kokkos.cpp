#include "integral_kokkos.h"

float IntegralKokkos(float start, float end, int count) {
    if (count <= 0) {
        return 0.0f;
    }

    const float step = (end - start) / static_cast<float>(count);
    float sum = 0.0f;

    Kokkos::parallel_reduce(
        "IntegralKokkos",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {count, count}),
        KOKKOS_LAMBDA(const int i, const int j, float& local_sum) {
            const float x = start + (static_cast<float>(i) + 0.5f) * step;
            const float y = start + (static_cast<float>(j) + 0.5f) * step;

            local_sum += Kokkos::sin(x) * Kokkos::cos(y);
        },
        sum
    );

    Kokkos::fence();

    return sum * step * step;
}
