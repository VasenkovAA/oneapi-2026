#include "acc_jacobi_oneapi.h"

#include <algorithm>
#include <vector>

constexpr size_t WORK_GROUP_SIZE = 64;

std::vector<float> JacobiAccONEAPI(
        const std::vector<float>& matrix,
        const std::vector<float>& rhs,
        float tolerance,
        sycl::device device) {

    const size_t num_equations = rhs.size();
    if (num_equations == 0u) return {};
    if (matrix.size() != num_equations * num_equations) return {};
    if (tolerance < 0.0f) tolerance = 0.0f;

    sycl::queue queue(device, sycl::property::queue::in_order{});

    std::vector<float> previous_solution(num_equations, 0.0f);
    std::vector<float> next_solution(num_equations, 0.0f);

    sycl::buffer<float, 1> mat_buf(matrix.data(), sycl::range<1>(matrix.size()));
    sycl::buffer<float, 1> rhs_buf(rhs.data(), sycl::range<1>(num_equations));
    sycl::buffer<float, 1> prev_buf(previous_solution.data(), sycl::range<1>(num_equations));
    sycl::buffer<float, 1> next_buf(next_solution.data(), sycl::range<1>(num_equations));

    float host_max_diff = 0.0f;
    sycl::buffer<float, 1> diff_buf(&host_max_diff, sycl::range<1>(1));

    const size_t num_blocks = (num_equations + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE;
    const size_t global_size = num_blocks * WORK_GROUP_SIZE;

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        queue.submit([&](sycl::handler& h) {
            auto diff_acc = diff_buf.get_access<sycl::access::mode::discard_write>(h);
            h.fill(diff_acc, 0.0f);
        }).wait();

        queue.submit([&](sycl::handler& h) {
            auto A = mat_buf.get_access<sycl::access::mode::read>(h);
            auto B = rhs_buf.get_access<sycl::access::mode::read>(h);
            auto X_prev = prev_buf.get_access<sycl::access::mode::read>(h);
            auto X_next = next_buf.get_access<sycl::access::mode::discard_write>(h);

            auto reduction = sycl::reduction(diff_buf, h, sycl::maximum<float>());

            h.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(WORK_GROUP_SIZE)),
                reduction,
                [=](sycl::nd_item<1> item, auto& max_reduce) {
                    const size_t i = item.get_global_id(0);
                    if (i >= num_equations) return;

                    const size_t row_start = i * num_equations;
                    const float diag = A[row_start + i];

                    if (sycl::fabs(diag) < 1e-12f) {
                        X_next[i] = X_prev[i];
                        max_reduce.combine(0.0f);
                        return;
                    }

                    float sum = 0.0f;
                    for (size_t j = 0; j < num_equations; ++j) {
                        if (j == i) continue;
                        sum += A[row_start + j] * X_prev[j];
                    }

                    const float new_value = (B[i] - sum) / diag;
                    X_next[i] = new_value;

                    const float diff = sycl::fabs(new_value - X_prev[i]);
                    max_reduce.combine(diff);
                }
            );
        }).wait();

        float current_max_diff = 0.0f;
        {
            sycl::host_accessor diff_acc(diff_buf, sycl::read_only);
            current_max_diff = diff_acc[0];
        }

        std::swap(previous_solution, next_solution);
        std::swap(prev_buf, next_buf);

        if (current_max_diff < tolerance) break;
    }

    return previous_solution;
}

