#include "block_gemm_oneapi.h"
#include <cmath>

std::vector<float> GemmBlockONEAPI(const std::vector<float> &a,
                                   const std::vector<float> &b, size_t size,
                                   sycl::device device) {

  sycl::queue gpu_queue(device);
  std::vector<float> res(size * size, 0.0f);
  const int block_size = std::min(size, 16);
  int num_blocks = size / block_size;

  auto alloc = [&gpu_queue](size_t size) -> float * {
    return sycl::malloc_device<float>(size, gpu_queue);
  };

  auto free = [&gpu_queue](void *ptr) -> void { sycl::free(ptr, gpu_queue); };

  float *in_a = alloc(size * size);
  float *in_b = alloc(size * size);
  float *in_c = alloc(size * size);

  int mem = size * size * sizeof(float);
  gpu_queue.memcpy(in_a, a.data(), mem);
  gpu_queue.memcpy(in_b, b.data(), mem);
  gpu_queue.memset(in_c, 0, mem);
  gpu_queue.wait();

  sycl::range<2> global(size, size);
  sycl::range<2> local(block_size, block_size);

  gpu_queue
      .submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 2> local_a(
            sycl::range<2>(block_size, block_size), cgh);
        sycl::local_accessor<float, 2> local_b(
            sycl::range<2>(block_size, block_size), cgh);

        cgh.parallel_for(sycl::nd_range<2>(global, local),
                         [=](sycl::nd_item<2> ids) {
                           int glob_i = ids.get_global_id(0);
                           int glob_j = ids.get_global_id(1);

                           int loc_i = ids.get_local_id(0);
                           int loc_j = ids.get_local_id(1);
                           float sum = 0.0f;

                           for (int k = 0; k < num_blocks; ++k) {
                             int i = k * block_size + loc_i;
                             int j = k * block_size + loc_j;

                             local_a[loc_i][loc_j] = in_a[glob_i * size + j];
                             local_b[loc_i][loc_j] = in_b[i * size + glob_j];

                             sycl::group_barrier(ids.get_group());

                             for (int k = 0; k < block_size; ++k) {
                               sum += local_a[loc_i][k] * local_b[k][loc_j];
                             }
                             sycl::group_barrier(ids.get_group());
                           }
                           in_c[glob_i * size + glob_j] = sum;
                         });
      })
      .wait();

  gpu_queue.memcpy(res.data(), in_c, mem).wait();

  free(in_a);
  free(in_b);
  free(in_c);

  return res;
}
