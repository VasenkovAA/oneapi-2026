#include "jacobi_kokkos.h"

std::vector<float> JacobiKokkos(
        const std::vector<float>& a,
        const std::vector<float>& b,
        float accuracy)
{
    if (accuracy <= 0.0f) {
        accuracy = 1e-6f;
    }

    const int N = static_cast<int>(b.size());
    if (N == 0 || a.size() != static_cast<size_t>(N) * N) {
        return {};
    }

    using ExecSpace = Kokkos::SYCL;
    using MemSpace = Kokkos::SYCLDeviceUSMSpace;
    using Layout = Kokkos::LayoutRight;

    Kokkos::View<float**, Layout, MemSpace> matrix("matrix", N, N);
    Kokkos::View<float*, MemSpace> rhs("rhs", N);
    Kokkos::View<float*, MemSpace> solution_curr("curr", N);
    Kokkos::View<float*, MemSpace> solution_next("next", N);
    Kokkos::View<float*, MemSpace> inv_diag("inv_diag", N);

    auto matrix_host = Kokkos::create_mirror_view(matrix);
    auto rhs_host = Kokkos::create_mirror_view(rhs);

    for (int i = 0; i < N; ++i) {
        rhs_host(i) = b[i];
        for (int j = 0; j < N; ++j) {
            matrix_host(i, j) = a[i * N + j];
        }
    }

    Kokkos::deep_copy(matrix, matrix_host);
    Kokkos::deep_copy(rhs, rhs_host);
    Kokkos::deep_copy(solution_curr, 0.0f);

    Kokkos::parallel_for(
        "init_inv_diag",
        Kokkos::RangePolicy<ExecSpace>(0, N),
        KOKKOS_LAMBDA(int i) {
            float diag = matrix(i, i);
            inv_diag(i) = (Kokkos::abs(diag) < 1e-12f) ? 1.0f : 1.0f / diag;
        });

    const int check_interval = 8;
    bool converged = false;

    for (int iteration = 0; iteration < ITERATIONS && !converged; ++iteration) {
        Kokkos::parallel_for(
            "jacobi_step",
            Kokkos::RangePolicy<ExecSpace>(0, N),
            KOKKOS_LAMBDA(int i) {
                float sum = 0.0f;
                for (int j = 0; j < N; ++j) {
                    if (j != i) {
                        sum += matrix(i, j) * solution_curr(j);
                    }
                }
                solution_next(i) = inv_diag(i) * (rhs(i) - sum);
            });

        if ((iteration + 1) % check_interval == 0) {
            float max_error = 0.0f;
            Kokkos::parallel_reduce(
                "check_convergence",
                Kokkos::RangePolicy<ExecSpace>(0, N),
                KOKKOS_LAMBDA(int i, float& local_max) {
                    float diff = Kokkos::abs(solution_next(i) - solution_curr(i));
                    if (diff > local_max) local_max = diff;
                },
                Kokkos::Max<float>(max_error));

            if (max_error < accuracy) {
                Kokkos::kokkos_swap(solution_curr, solution_next);
                converged = true;
                break;
            }
        }

        Kokkos::kokkos_swap(solution_curr, solution_next);
    }

    auto result_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), solution_curr);
    std::vector<float> result(N);
    for (int i = 0; i < N; ++i) {
        result[i] = result_host(i);
    }

    return result;
}