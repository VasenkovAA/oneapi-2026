#include "jacobi_kokkos.h"
#include <algorithm>
#include <cmath>

using namespace Kokkos;

std::vector<float> JacobiKokkos(const std::vector<float>& a,
                                const std::vector<float>& b,
                                float accuracy) {

  int size = b.size();

  View<float **, LayoutLeft, SYLDeviceUSMSpace> in_a("in_a", size, size);
  View<float *, SYCLDeviceUSMSpace> in_b("in_b", size);
  View<float *, SYCLDeviceUSMSpace> prev_res("prev_res", size);
  View<float *, SYCLDeviceUSMSpace> res("res", size);

  auto host_a = create_mirror_view(in_a);
  auto host_b = create_mirror_view(in_b);

  for (int i = 0; i < size; ++i) {
    host_b(i) = b[i];
    for (int indx = 0; indx < size; ++indx) {
      host_a(i, indx) = a[i * size + indx];
    }
  }

  deep_copy(in_a, host_a);
  deep_copy(in_b, host_b);
  deep_copy(prev_res, 0.0f);
  deep_copy(res, 0.0f);

  for (int epoch = 0; epoch < ITERATIONS; ++epoch) {

    parallel_for(
        "Jacobi", RangePolicy<SYCL>(0, size), KOKKOS_LAMBDA(int idx) {
          float next_res = 0.0f;
          for (int indx = 0; indx < size; ++indx) {
            if (indx != idx) {
              next_res += in_a(idx, indx) * prev_res(indx);
            }
          }
          res(idx) = (in_b(idx) - next_res) / in_a(idx, idx);
        });

    fence();

    float norm = 0.0f;
    parallel_reduce(
        "Check", RangePolicy<SYCL>(0, size),
        KOKKOS_LAMBDA(int i, float &mx) {
          float el = std::fabs(res(i) - prev_res(i));
          if (el > mx)
            mx = el;
        },
        Kokkos::Max<float>(norm));

    deep_copy(prev_res, res);

    if (norm < accuracy) {
      break;
    }
  }

  std::vector<float> result(size);
  auto host_prev_res = create_mirror_view_and_copy(HostSpace(), prev_res);
  for (int idx = 0; idx < size; ++idx) {
    result[idx] = host_prev_res(idx);
  }

  return result;
}
