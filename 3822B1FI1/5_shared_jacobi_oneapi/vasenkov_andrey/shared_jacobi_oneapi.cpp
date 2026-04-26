#include "shared_jacobi_oneapi.h"

#include <cmath>
#include <algorithm>
#include <vector>

std::vector<float> JacobiSharedONEAPI(
        const std::vector<float>& matrix,
        const std::vector<float>& rhs,
        float accuracy,
        sycl::device device)
{
    if (accuracy <= 0.0f) {
        accuracy = 1e-6f;
    }

    const size_t size = rhs.size();
    if (size == 0 || matrix.size() != size * size) {
        return {};
    }

    try {
        sycl::queue queue{device};

        float* d_matrix = sycl::malloc_shared<float>(matrix.size(), queue);
        float* d_rhs = sycl::malloc_shared<float>(size, queue);
        float* d_x_old = sycl::malloc_shared<float>(size, queue);
        float* d_x_new = sycl::malloc_shared<float>(size, queue);
        float* d_inv_diag = sycl::malloc_shared<float>(size, queue);
        float* d_max_diff = sycl::malloc_shared<float>(1, queue);

        if (!d_matrix || !d_rhs || !d_x_old || !d_x_new || !d_inv_diag || !d_max_diff) {
            sycl::free(d_matrix, queue);
            sycl::free(d_rhs, queue);
            sycl::free(d_x_old, queue);
            sycl::free(d_x_new, queue);
            sycl::free(d_inv_diag, queue);
            sycl::free(d_max_diff, queue);
            return {};
        }

        std::copy(matrix.begin(), matrix.end(), d_matrix);
        std::copy(rhs.begin(), rhs.end(), d_rhs);
        std::fill(d_x_old, d_x_old + size, 0.0f);
        std::copy(d_x_old, d_x_old + size, d_x_new);
        *d_max_diff = 0.0f;

        for (size_t i = 0; i < size; ++i) {
            float diag = d_matrix[i * size + i];
            d_inv_diag[i] = (std::fabs(diag) > 1e-12f) ? (1.0f / diag) : 0.0f;
        }

        for (int iter = 0; iter < ITERATIONS; ++iter)
        {
            *d_max_diff = 0.0f;

            queue.submit([&](sycl::handler& handler)
            {
                auto reduction = sycl::reduction(d_max_diff, handler, sycl::maximum<float>());

                handler.parallel_for(
                    sycl::range<1>{size},
                    reduction,
                    [=](sycl::id<1> id, auto& local_max)
                    {
                        const size_t i = id[0];
                        float sigma = 0.0f;

                        for (size_t j = 0; j < size; ++j)
                        {
                            if (j != i)
                            {
                                sigma += d_matrix[i * size + j] * d_x_old[j];
                            }
                        }

                        float inv_diag = d_inv_diag[i];
                        float new_val = (inv_diag == 0.0f) ? d_x_old[i] : (d_rhs[i] - sigma) * inv_diag;
                        d_x_new[i] = new_val;

                        float diff = sycl::fabs(new_val - d_x_old[i]);
                        local_max.combine(diff);
                    });
            }).wait();

            std::swap(d_x_old, d_x_new);

            if (*d_max_diff < accuracy)
            {
                break;
            }
        }

        std::vector<float> solution(size);
        std::copy(d_x_old, d_x_old + size, solution.begin());

        sycl::free(d_matrix, queue);
        sycl::free(d_rhs, queue);
        sycl::free(d_x_old, queue);
        sycl::free(d_x_new, queue);
        sycl::free(d_inv_diag, queue);
        sycl::free(d_max_diff, queue);

        return solution;
    }
    catch (sycl::exception const&)
    {
        return {};
    }
}