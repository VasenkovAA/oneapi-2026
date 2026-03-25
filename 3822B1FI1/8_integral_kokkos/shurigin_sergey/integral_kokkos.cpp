#include "integral_kokkos.h"

float IntegralKokkos(float start, float end, int count) {
    float h = (end - start) / count;
    float total_sum = 0.0f;

    auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {count, count});

    Kokkos::parallel_reduce("DoubleIntegral", policy, 
        KOKKOS_LAMBDA (const int i, const int j, float& lsum) {
            float x = start + (i + 0.5f) * h;
            float y = start + (j + 0.5f) * h;
            lsum += Kokkos::sin(x) * Kokkos::cos(y);
        }, total_sum);

    return total_sum * h * h;
}