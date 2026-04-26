#include "block_gemm_oneapi.h"

std::vector<float> GemmBlockONEAPI(const std::vector<float>& a,
                                   const std::vector<float>& b,
                                   size_t size,
                                   sycl::device device) {
  constexpr size_t TILE_SIZE = 16;
  const size_t blockSize = (size < TILE_SIZE) ? size : TILE_SIZE;

  const size_t total = size * size;
  std::vector<float> result(total);

  sycl::queue q(device, sycl::property::queue::in_order{});

  float* dA = sycl::malloc_device<float>(total, q);
  float* dB = sycl::malloc_device<float>(total, q);
  float* dC = sycl::malloc_device<float>(total, q);

  q.memcpy(dA, a.data(), total * sizeof(float));
  q.memcpy(dB, b.data(), total * sizeof(float));
  q.fill(dC, 0.0f, total);

  const size_t numBlocks = (size + blockSize - 1) / blockSize;
  const size_t globalSize = numBlocks * blockSize;

  sycl::range<2> block(blockSize, blockSize);
  sycl::range<2> grid(globalSize, globalSize);

  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 2> localA(block, h);
    sycl::local_accessor<float, 2> localB(block, h);

    h.parallel_for(sycl::nd_range<2>(grid, block), [=](sycl::nd_item<2> it) {
      const size_t row = it.get_global_id(0);
      const size_t col = it.get_global_id(1);
      const size_t lrow = it.get_local_id(0);
      const size_t lcol = it.get_local_id(1);

      float sum = 0.0f;

      for (size_t blk = 0; blk < numBlocks; ++blk) {
        const size_t loadCol = blk * blockSize + lcol;
        const size_t loadRow = blk * blockSize + lrow;

        if (row < size && loadCol < size)
          localA[lrow][lcol] = dA[row * size + loadCol];
        else
          localA[lrow][lcol] = 0.0f;

        if (loadRow < size && col < size)
          localB[lrow][lcol] = dB[loadRow * size + col];
        else
          localB[lrow][lcol] = 0.0f;

        sycl::group_barrier(it.get_group());

        for (size_t k = 0; k < blockSize; ++k)
          sum += localA[lrow][k] * localB[k][lcol];

        sycl::group_barrier(it.get_group());
      }

      if (row < size && col < size)
        dC[row * size + col] = sum;
    });
  });

  q.memcpy(result.data(), dC, total * sizeof(float)).wait();

  sycl::free(dA, q);
  sycl::free(dB, q);
  sycl::free(dC, q);

  return result;
}