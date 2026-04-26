#include "jacobi_kokkos.h"

#include <cmath>

std::vector<float> JacobiKokkos(
    const std::vector<float>& a,
    const std::vector<float>& b,
    float accuracy
) {
    const int n = b.size();

    using View = Kokkos::View<float*>;

    View a_view("a", n * n);
    View b_view("b", n);
    View x("x", n);
    View x_new("x_new", n);

    auto a_host = Kokkos::create_mirror_view(a_view);
    auto b_host = Kokkos::create_mirror_view(b_view);

    for (int i = 0; i < n * n; i++) {
        a_host(i) = a[i];
    }
    for (int i = 0; i < n; i++) {
        b_host(i) = b[i];
    }

    Kokkos::deep_copy(a_view, a_host);
    Kokkos::deep_copy(b_view, b_host);

    Kokkos::deep_copy(x, 0.0f);
    Kokkos::deep_copy(x_new, 0.0f);

    for (int iter = 0; iter < ITERATIONS; iter++) {

        Kokkos::parallel_for(
            "JacobiCompute",
            Kokkos::RangePolicy<>(0, n),
            KOKKOS_LAMBDA(const int i) {
            float s = 0.0f;

            for (int j = 0; j < n; j++) {
                if (j != i) {
                    s += a_view(i * n + j) * x(j);
                }
            }

            x_new(i) = (b_view(i) - s) / a_view(i * n + i);
        }
        );

        float diff = 0.0f;

        Kokkos::parallel_reduce(
            "JacobiDiff",
            Kokkos::RangePolicy<>(0, n),
            KOKKOS_LAMBDA(const int i, float& local_sum) {
            float d = x_new(i) - x(i);
            local_sum += d * d;
        },
            diff
        );

        if (std::sqrt(diff) < accuracy) {
            break;
        }

        Kokkos::parallel_for(
            "JacobiSwap",
            Kokkos::RangePolicy<>(0, n),
            KOKKOS_LAMBDA(const int i) {
            x(i) = x_new(i);
        }
        );
    }

    auto x_host = Kokkos::create_mirror_view(x);
    Kokkos::deep_copy(x_host, x);

    std::vector<float> result(n);
    for (int i = 0; i < n; i++) {
        result[i] = x_host(i);
    }

    return result;
}