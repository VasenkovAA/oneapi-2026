#include "block_gemm_oneapi.h"

std::vector<float> GemmBlockONEAPI(
    const std::vector<float>& a, const std::vector<float>& b,
    size_t size, sycl::device device)
{
    const size_t BLOCK_SIZE = 16;
    std::vector<float> c(size * size, 0.0f);

    sycl::queue queue(device);

    float* d_a = sycl::malloc_device<float>(size * size, queue);
    float* d_b = sycl::malloc_device<float>(size * size, queue);
    float* d_c = sycl::malloc_device<float>(size * size, queue);

    queue.memcpy(d_a, a.data(), size * size * sizeof(float));
    queue.memcpy(d_b, b.data(), size * size * sizeof(float));
    queue.memcpy(d_c, c.data(), size * size * sizeof(float));
    queue.wait();

    size_t block_count = size / BLOCK_SIZE;

    queue.submit([&](sycl::handler& cgh)
        {
            sycl::local_accessor<float, 2> a_tile(sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);
            sycl::local_accessor<float, 2> b_tile(sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);

            cgh.parallel_for(
                sycl::nd_range<2>(
                    sycl::range<2>(block_count * BLOCK_SIZE, block_count * BLOCK_SIZE),
                    sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE)
                ),
                [=](sycl::nd_item<2> item)
                {
                    size_t row = item.get_global_id(0);
                    size_t col = item.get_global_id(1);
                    size_t local_row = item.get_local_id(0);
                    size_t local_col = item.get_local_id(1);

                    float sum = 0.0f;

                    for (size_t k = 0; k < block_count; ++k)
                    {
                        a_tile[local_row][local_col] = d_a[row * size + k * BLOCK_SIZE + local_col];
                        b_tile[local_row][local_col] = d_b[(k * BLOCK_SIZE + local_row) * size + col];

                        item.barrier(sycl::access::fence_space::local_space);

                        for (size_t i = 0; i < BLOCK_SIZE; ++i)
                        {
                            sum += a_tile[local_row][i] * b_tile[i][local_col];
                        }

                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    d_c[row * size + col] = sum;
                });
        });
    queue.wait();

    queue.memcpy(c.data(), d_c, size * size * sizeof(float));
    queue.wait();

    sycl::free(d_a, queue);
    sycl::free(d_b, queue);
    sycl::free(d_c, queue);

    return c;
}