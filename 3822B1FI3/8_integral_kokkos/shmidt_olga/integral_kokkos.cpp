#include "integral_kokkos.h"
#include <cmath>

float IntegralKokkos(float start, float end, int count)
{
    float step = (end - start) / count;
    float total_sum = 0.0f;
    
    Kokkos::parallel_reduce(
        "double_integral",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {count, count}),
        KOKKOS_LAMBDA(int i, int j, float& sum)
        {
            float x = start + (i + 0.5f) * step;
            float y = start + (j + 0.5f) * step;
            sum += sinf(x) * cosf(y) * step * step;
        },
        total_sum);
    
    return total_sum;
}
