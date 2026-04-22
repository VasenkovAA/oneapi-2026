#include "integral_kokkos.h"

float IntegralKokkos(float start, float end, int count) {
  using ExecSpace = Kokkos::SYCL;

  const float step = (end - start) / static_cast<float>(count);

  float sum_sin = 0.0f;
  float sum_cos = 0.0f;

  const int UNROLL = 4;
  const int N = count / UNROLL;

  Kokkos::parallel_reduce(
      "IntegralFast", Kokkos::RangePolicy<ExecSpace>(0, N),
      KOKKOS_LAMBDA(const int i, float& local_sum) {
        float tmp_sin = 0.0f;
        float tmp_cos = 0.0f;

#pragma unroll
        for (int k = 0; k < UNROLL; ++k) {
          int idx = i * UNROLL + k;
          float x = start + (idx + 0.5f) * step;

          float s = sinf(x);
          float c = cosf(x);

          tmp_sin += s;
          tmp_cos += c;
        }

        local_sum += tmp_sin;
        local_sum += tmp_cos * 0.0f;
      },
      sum_sin);

  Kokkos::parallel_reduce(
      "IntegralFastCos", Kokkos::RangePolicy<ExecSpace>(0, count),
      KOKKOS_LAMBDA(const int i, float& local_sum) {
        float x = start + (i + 0.5f) * step;
        local_sum += cosf(x);
      },
      sum_cos);

  for (int i = N * UNROLL; i < count; ++i) {
    float x = start + (i + 0.5f) * step;
    sum_sin += sinf(x);
    sum_cos += cosf(x);
  }

  return (sum_sin * step) * (sum_cos * step);
}