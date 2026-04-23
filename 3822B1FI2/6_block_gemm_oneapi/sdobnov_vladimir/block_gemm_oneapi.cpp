#include "block_gemm_oneapi.h"

#include <cmath>

std::vector<float> GemmBlockONEAPI(
    const std::vector<float>& a,
    const std::vector<float>& b,
    size_t size,
    sycl::device device
) {
    sycl::queue q(device);

    const size_t n = size;
    const size_t BLOCK = 16;

    float* a_mem = sycl::malloc_shared<float>(n * n, q);
    float* b_mem = sycl::malloc_shared<float>(n * n, q);
    float* c_mem = sycl::malloc_shared<float>(n * n, q);

    for (size_t i = 0; i < n * n; i++) {
        a_mem[i] = a[i];
        b_mem[i] = b[i];
        c_mem[i] = 0.0f;
    }

    sycl::range<2> global(
        (n + BLOCK - 1) / BLOCK * BLOCK,
        (n + BLOCK - 1) / BLOCK * BLOCK
    );

    sycl::range<2> local(BLOCK, BLOCK);

    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 2> tileA({ BLOCK, BLOCK }, h);
        sycl::local_accessor<float, 2> tileB({ BLOCK, BLOCK }, h);

        h.parallel_for(
            sycl::nd_range<2>(global, local),
            [=](sycl::nd_item<2> item) {
                size_t row = item.get_global_id(0);
                size_t col = item.get_global_id(1);

                size_t local_row = item.get_local_id(0);
                size_t local_col = item.get_local_id(1);

                float sum = 0.0f;

                for (size_t t = 0; t < n; t += BLOCK) {
                    if (row < n && (t + local_col) < n) {
                        tileA[local_row][local_col] =
                            a_mem[row * n + (t + local_col)];
                    }
                    else {
                        tileA[local_row][local_col] = 0.0f;
                    }

                    if (col < n && (t + local_row) < n) {
                        tileB[local_row][local_col] =
                            b_mem[(t + local_row) * n + col];
                    }
                    else {
                        tileB[local_row][local_col] = 0.0f;
                    }

                    item.barrier(sycl::access::fence_space::local_space);

                    for (size_t k = 0; k < BLOCK; k++) {
                        sum += tileA[local_row][k] * tileB[k][local_col];
                    }

                    item.barrier(sycl::access::fence_space::local_space);
                }

                if (row < n && col < n) {
                    c_mem[row * n + col] = sum;
                }
            }
        );
        });

    q.wait();

    std::vector<float> result(n * n);
    for (size_t i = 0; i < n * n; i++) {
        result[i] = c_mem[i];
    }

    sycl::free(a_mem, q);
    sycl::free(b_mem, q);
    sycl::free(c_mem, q);

    return result;
}