#include "integral_kokkos.h"

float IntegralKokkos(float lowerBound, float upperBound, int intervals) {
    if (intervals <= 0) {
        return 0.0f;
    }

    using ExecutionSpace = Kokkos::SYCL;
    using MemorySpace = Kokkos::SYCLDeviceUSMSpace;

    const float step = (upperBound - lowerBound) / static_cast<float>(intervals);
    const float area = step * step;

    float total = 0.0f;

    Kokkos::parallel_reduce(
        "DoubleIntegral",
        Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<2>>(
            {{0, 0}},
            {{intervals, intervals}}
        ),
        KOKKOS_LAMBDA(const int i, const int j, float& tempSum) {
            const float x = lowerBound + (static_cast<float>(i) + 0.5f) * step;
            const float y = lowerBound + (static_cast<float>(j) + 0.5f) * step;
            tempSum += Kokkos::sin(x) * Kokkos::cos(y);
        },
        total
    );

    return total * area;
}