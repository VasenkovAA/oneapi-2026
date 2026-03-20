#include "dev_jacobi_oneapi.h"
#include <algorithm>
#include <cmath>

std::vector<float> JacobiDevONEAPI(const std::vector<float> &matrix_a,
                                   const std::vector<float> &vector_b,
                                   float accuracy, sycl::device device) {

  size_t matrix_size = vector_b.size();
  std::vector<float> current_solution(matrix_size, 0.0f);
  std::vector<float> next_solution(matrix_size, 0.0f);

  sycl::queue computation_queue(device);

  float *device_matrix =
      sycl::malloc_device<float>(matrix_size * matrix_size, computation_queue);
  float *device_rhs =
      sycl::malloc_device<float>(matrix_size, computation_queue);
  float *device_current =
      sycl::malloc_device<float>(matrix_size, computation_queue);
  float *device_next =
      sycl::malloc_device<float>(matrix_size, computation_queue);

  computation_queue
      .memcpy(device_matrix, matrix_a.data(),
              sizeof(float) * matrix_size * matrix_size)
      .wait();
  computation_queue
      .memcpy(device_rhs, vector_b.data(), sizeof(float) * matrix_size)
      .wait();
  computation_queue.memset(device_current, 0, sizeof(float) * matrix_size)
      .wait();
  computation_queue.memset(device_next, 0, sizeof(float) * matrix_size).wait();

  std::vector<float> diagonal_elements(matrix_size);
  for (size_t i = 0; i < matrix_size; ++i) {
    diagonal_elements[i] = matrix_a[i * matrix_size + i];
  }
  float *device_diagonal =
      sycl::malloc_device<float>(matrix_size, computation_queue);
  computation_queue
      .memcpy(device_diagonal, diagonal_elements.data(),
              sizeof(float) * matrix_size)
      .wait();

  std::vector<float> current_host(matrix_size);
  std::vector<float> next_host(matrix_size);

  for (int iteration = 0; iteration < MAX_ITERATIONS; ++iteration) {
    computation_queue
        .parallel_for(
            sycl::range<1>(matrix_size),
            [=](sycl::id<1> element_id) {
              size_t row = element_id[0];
              float summation = 0.0f;

              for (size_t column = 0; column < matrix_size; ++column) {
                if (column != row) {
                  summation += device_matrix[row * matrix_size + column] *
                               device_current[column];
                }
              }

              device_next[row] =
                  (device_rhs[row] - summation) / device_diagonal[row];
            })
        .wait();

    computation_queue
        .memcpy(current_host.data(), device_current,
                sizeof(float) * matrix_size)
        .wait();
    computation_queue
        .memcpy(next_host.data(), device_next, sizeof(float) * matrix_size)
        .wait();

    float max_difference = 0.0f;
    for (size_t i = 0; i < matrix_size; ++i) {
      max_difference =
          std::max(max_difference, std::fabs(next_host[i] - current_host[i]));
    }

    if (max_difference < accuracy) {
      break;
    }

    computation_queue
        .memcpy(device_current, device_next, sizeof(float) * matrix_size)
        .wait();
  }

  std::vector<float> final_solution(matrix_size);
  computation_queue
      .memcpy(final_solution.data(), device_next, sizeof(float) * matrix_size)
      .wait();

  sycl::free(device_matrix, computation_queue);
  sycl::free(device_rhs, computation_queue);
  sycl::free(device_current, computation_queue);
  sycl::free(device_next, computation_queue);
  sycl::free(device_diagonal, computation_queue);

  return final_solution;
}