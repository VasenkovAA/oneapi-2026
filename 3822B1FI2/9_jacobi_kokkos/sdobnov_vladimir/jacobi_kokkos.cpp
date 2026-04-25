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

    // Копирование данных
    for (int i = 0; i < n * n; i++) {
        a_view(i) = a[i];
    }
    for (int i = 0; i < n; i++) {
        b_view(i) = b[i];
        x(i) = 0.0f;
        x_new(i) = 0.0f;
    }

    for (int iter = 0; iter < ITERATIONS; iter++) {

        // 1. Вычисление x_new
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

        // 2. Вычисление нормы
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

        // 3. swap
        Kokkos::parallel_for(
            "JacobiSwap",
            Kokkos::RangePolicy<>(0, n),
            KOKKOS_LAMBDA(const int i) {
            x(i) = x_new(i);
        }
        );
    }

    std::vector<float> result(n);

    for (int i = 0; i < n; i++) {
        result[i] = x(i);
    }

    return result;
}