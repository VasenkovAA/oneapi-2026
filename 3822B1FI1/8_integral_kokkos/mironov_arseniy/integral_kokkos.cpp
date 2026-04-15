#include "integral_kokkos.h"
#include <Kokkos_MathematicalFunctions.hpp>

float IntegralKokkos(float start, float end, int count) {

  float scale = (end - start) / count;
  float res = 0.0f;

  Kokkos::parallel_reduce(
      Kokkos::MDRangePolicy<Kokkos::SYCL, Kokkos::Rank<2>>({0, 0},
                                                           {count, count}),
      KOKKOS_LAMBDA(int i, int j, float &sum) {
        float x = start + (i + 0.5f) * scale;
        float y = start + (j + 0.5f) * scale;
        sum += Kokkos::sin(x) * Kokkos::cos(y);
      },
      res);

  return res * scale * scale;
}
