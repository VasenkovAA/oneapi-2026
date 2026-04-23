#include "jacobi_kokkos.h"

std::vector<float> JacobiKokkos(const std::vector<float>& a,
                                const std::vector<float>& b, float accuracy) {
  using ExecSpace = Kokkos::SYCL;
  using MemSpace = Kokkos::SYCLDeviceUSMSpace;

  const int n = static_cast<int>(b.size());

  Kokkos::View<float*, MemSpace> A("A", n * n);
  Kokkos::View<float*, MemSpace> B("B", n);
  Kokkos::View<float*, MemSpace> x_old("x_old", n);
  Kokkos::View<float*, MemSpace> x_new("x_new", n);
  Kokkos::View<float*, MemSpace> inv_diag("inv_diag", n);

  auto hA = Kokkos::create_mirror_view(A);
  auto hB = Kokkos::create_mirror_view(B);

  for (int i = 0; i < n * n; ++i) hA(i) = a[i];
  for (int i = 0; i < n; ++i) hB(i) = b[i];

  Kokkos::deep_copy(A, hA);
  Kokkos::deep_copy(B, hB);
  Kokkos::deep_copy(x_old, 0.0f);

  Kokkos::parallel_for(
      "InitInvDiag", Kokkos::RangePolicy<ExecSpace>(0, n),
      KOKKOS_LAMBDA(int i) { inv_diag(i) = 1.0f / A(i * n + i); });

  for (int iter = 0; iter < ITERATIONS; ++iter) {
    float error = 0.0f;

    Kokkos::parallel_reduce(
        "JacobiIteration", Kokkos::RangePolicy<ExecSpace>(0, n),
        KOKKOS_LAMBDA(int i, float& max_err) {
          const float* row = &A(i * n);
          const float xi = x_old(i);

          double sigma = 0.0;

          int j = 0;
          for (; j + 4 <= n; j += 4) {
            sigma += (double)row[j] * x_old(j);
            sigma += (double)row[j + 1] * x_old(j + 1);
            sigma += (double)row[j + 2] * x_old(j + 2);
            sigma += (double)row[j + 3] * x_old(j + 3);
          }
          for (; j < n; ++j) {
            sigma += (double)row[j] * x_old(j);
          }

          sigma -= (double)row[i] * xi;

          const float new_val =
              (B(i) - static_cast<float>(sigma)) * inv_diag(i);

          const float diff = fabsf(new_val - xi);

          x_new(i) = new_val;

          if (diff > max_err) max_err = diff;
        },
        Kokkos::Max<float>(error));

    ExecSpace().fence();

    Kokkos::kokkos_swap(x_old, x_new);

    if (error < accuracy) break;
  }

  auto hX = Kokkos::create_mirror_view(x_old);
  Kokkos::deep_copy(hX, x_old);

  std::vector<float> result(n);
  for (int i = 0; i < n; ++i) result[i] = hX(i);

  return result;
}