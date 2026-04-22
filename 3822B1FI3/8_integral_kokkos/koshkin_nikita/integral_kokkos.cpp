#include "integral_kokkos.h"

float IntegralKokkos(float start, float end, int count) {
  const float step = (end - start) / static_cast<float>(count);
  float result = 0.0f;

  Kokkos::parallel_reduce(
      Kokkos::MDRangePolicy<Kokkos::SYCL, Kokkos::Rank<2>>({0, 0},
                                                           {count, count}),
      KOKKOS_LAMBDA(int i, int j, float& sum) {
        const float x = start + (i + 0.5f) * step;
        const float y = start + (j + 0.5f) * step;

        sum += Kokkos::sin(x) * Kokkos::cos(y);
      },
      result);

  return result * step * step;
}