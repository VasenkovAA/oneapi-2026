#include "integral_kokkos.h"
#include <cmath>

float IntegralKokkos(float start, float end, int count) {
    if (count <= 0) {
        return 0.0f;
    }

    const float h = (end - start) / count;

    float sum = 0.0f;

    Kokkos::parallel_reduce(
        "Integral2D",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({ 0, 0 }, { count, count }),
        KOKKOS_LAMBDA(int i, int j, float& local_sum) {

        float x = start + (i + 0.5f) * h;
        float y = start + (j + 0.5f) * h;

        local_sum += sinf(x) * cosf(y);
    },
        sum
    );

    Kokkos::fence();

    return sum * h * h;
}