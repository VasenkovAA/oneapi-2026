#include "block_gemm_oneapi.h"

std::vector<float> GemmBlockONEAPI(const std::vector<float>& a,
                                   const std::vector<float>& b, size_t size,
                                   sycl::device device) {
  constexpr size_t block_size = 16;
  const size_t total_size = size * size;

  std::vector<float> result(total_size, 0.0f);

  sycl::queue queue(device);

  float* a_dev = sycl::malloc_device<float>(total_size, queue);
  float* b_dev = sycl::malloc_device<float>(total_size, queue);
  float* c_dev = sycl::malloc_device<float>(total_size, queue);

  queue.memcpy(a_dev, a.data(), total_size * sizeof(float));
  queue.memcpy(b_dev, b.data(), total_size * sizeof(float));
  queue.memset(c_dev, 0, total_size * sizeof(float)).wait();

  const size_t global_rows =
      ((size + block_size - 1) / block_size) * block_size;
  const size_t global_cols =
      ((size + block_size - 1) / block_size) * block_size;

  sycl::range<2> global_range(global_rows, global_cols);
  sycl::range<2> local_range(block_size, block_size);

  queue
      .submit([&](sycl::handler& handler) {
        sycl::local_accessor<float, 2> a_block(local_range, handler);
        sycl::local_accessor<float, 2> b_block(local_range, handler);

        handler.parallel_for(
            sycl::nd_range<2>(global_range, local_range),
            [=](sycl::nd_item<2> item) {
              const size_t row = item.get_global_id(0);
              const size_t col = item.get_global_id(1);

              const size_t local_row = item.get_local_id(0);
              const size_t local_col = item.get_local_id(1);

              float sum = 0.0f;

              for (size_t block_begin = 0; block_begin < size;
                   block_begin += block_size) {
                const size_t a_col = block_begin + local_col;
                const size_t b_row = block_begin + local_row;

                a_block[local_row][local_col] =
                    (row < size && a_col < size) ? a_dev[row * size + a_col]
                                                : 0.0f;

                b_block[local_row][local_col] =
                    (b_row < size && col < size) ? b_dev[b_row * size + col]
                                                : 0.0f;

                item.barrier(sycl::access::fence_space::local_space);

                for (size_t k = 0; k < block_size; ++k) {
                  sum += a_block[local_row][k] * b_block[k][local_col];
                }

                item.barrier(sycl::access::fence_space::local_space);
              }

              if (row < size && col < size) {
                c_dev[row * size + col] = sum;
              }
            });
      })
      .wait();

  queue.memcpy(result.data(), c_dev, total_size * sizeof(float)).wait();

  sycl::free(a_dev, queue);
  sycl::free(b_dev, queue);
  sycl::free(c_dev, queue);

  return result;
}