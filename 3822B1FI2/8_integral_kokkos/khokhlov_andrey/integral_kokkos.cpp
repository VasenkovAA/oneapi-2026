#include "integral_kokkos.h"
#include <cmath>

float IntegralKokkos(float start, float end, int count) {
    const float step_size = (end - start) / static_cast<float>(count);
    float total_sum = 0.0f;
    
    Kokkos::parallel_reduce(
        "Integral",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {count, count}),
        KOKKOS_LAMBDA(int idx_x, int idx_y, float& accumulator) {
            float midpoint_x = start + step_size * (static_cast<float>(idx_x) + 0.5f);
            float midpoint_y = start + step_size * (static_cast<float>(idx_y) + 0.5f);
            accumulator += std::sin(midpoint_x) * std::cos(midpoint_y);
        },
        total_sum
    );
    
    return total_sum * step_size * step_size;
}