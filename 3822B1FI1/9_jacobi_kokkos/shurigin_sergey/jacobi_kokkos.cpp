#include "jacobi_kokkos.h"

std::vector<float> JacobiKokkos(const std::vector<float>& a, const std::vector<float>& b, float accuracy) {
    int n = static_cast<int>(b.size());
    Kokkos::View<float**> v_a("a", n, n);
    Kokkos::View<float*> v_b("b", n);
    Kokkos::View<float*> v_x("x", n);
    Kokkos::View<float*> v_x_new("x_new", n);
    Kokkos::View<float*> v_inv_diag("inv", n);

    auto h_a = Kokkos::create_mirror_view(v_a);
    auto h_b = Kokkos::create_mirror_view(v_b);
    for (int i = 0; i < n; ++i) {
        h_b(i) = b[i];
        for (int j = 0; j < n; ++j) h_a(i, j) = a[i * n + j];
    }
    Kokkos::deep_copy(v_a, h_a);
    Kokkos::deep_copy(v_b, h_b);
    Kokkos::deep_copy(v_x, 0.0f);

    Kokkos::parallel_for("Init", n, KOKKOS_LAMBDA(const int i) {
        v_inv_diag(i) = 1.0f / v_a(i, i);
    });

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        Kokkos::parallel_for("JacobiStep", n, KOKKOS_LAMBDA(const int i) {
            float sum = 0.0f;
            for (int j = 0; j < n; ++j) {
                if (i != j) sum += v_a(i, j) * v_x(j);
            }
            v_x_new(i) = (v_b(i) - sum) * v_inv_diag(i);
        });

        float max_diff = 0.0f;
        Kokkos::parallel_reduce("JacobiCheck", n, KOKKOS_LAMBDA(const int i, float& lmax) {
            float diff = Kokkos::abs(v_x_new(i) - v_x(i));
            if (diff > lmax) lmax = diff;
        }, Kokkos::Max<float>(max_diff));

        Kokkos::kokkos_swap(v_x, v_x_new);
        if (max_diff < accuracy) break;
    }

    std::vector<float> result(n);
    auto h_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), v_x);
    for (int i = 0; i < n; ++i) result[i] = h_x(i);
    return result;
}