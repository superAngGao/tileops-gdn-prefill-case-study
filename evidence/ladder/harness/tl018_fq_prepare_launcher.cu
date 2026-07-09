#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <tl_templates/cuda/common.h>

#include <cstdint>
#include <stdexcept>

extern "C" __global__ void tilelang_kkt_solve_kernel_kernel(
    half_t* __restrict__ a,
    __grid_constant__ const CUtensorMap a_desc,
    const half_t* __restrict__ b,
    __grid_constant__ const CUtensorMap k_desc,
    int data_batch_size,
    int num_chunks,
    int num_tokens);

namespace {

void check_cuda(CUresult result, const char* what) {
  if (result != CUDA_SUCCESS) {
    const char* name = nullptr;
    const char* text = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &text);
    throw std::runtime_error(std::string(what) + ": " + (name ? name : "CUDA_ERROR") +
                             " - " + (text ? text : ""));
  }
}

void make_a_desc(CUtensorMap* desc, void* ptr, int batch, int tokens) {
  const uint64_t global_shape[4] = {64, 16, static_cast<uint64_t>(tokens),
                                    static_cast<uint64_t>(batch)};
  const uint64_t global_stride[3] = {128, 2048,
                                     static_cast<uint64_t>(tokens) * 2048ULL};
  const uint32_t box_dim[4] = {64, 1, 64, 1};
  const uint32_t elem_stride[4] = {1, 1, 1, 1};
  check_cuda(cuTensorMapEncodeTiled(
                 desc,
                 CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
                 4,
                 ptr,
                 global_shape,
                 global_stride,
                 box_dim,
                 elem_stride,
                 CU_TENSOR_MAP_INTERLEAVE_NONE,
                 CU_TENSOR_MAP_SWIZZLE_128B,
                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                 CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
             "cuTensorMapEncodeTiled(A)");
}

void make_k_desc(CUtensorMap* desc, const void* ptr, int batch, int tokens) {
  const uint64_t global_shape[4] = {128, 16, static_cast<uint64_t>(tokens),
                                    static_cast<uint64_t>(batch)};
  const uint64_t global_stride[3] = {256, 4096,
                                     static_cast<uint64_t>(tokens) * 4096ULL};
  const uint32_t box_dim[4] = {64, 1, 64, 1};
  const uint32_t elem_stride[4] = {1, 1, 1, 1};
  check_cuda(cuTensorMapEncodeTiled(
                 desc,
                 CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
                 4,
                 const_cast<void*>(ptr),
                 global_shape,
                 global_stride,
                 box_dim,
                 elem_stride,
                 CU_TENSOR_MAP_INTERLEAVE_NONE,
                 CU_TENSOR_MAP_SWIZZLE_128B,
                 CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                 CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
             "cuTensorMapEncodeTiled(K)");
}

void check_tensor(const torch::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(t.scalar_type() == torch::kFloat16, name, " must be float16");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

}  // namespace

torch::Tensor tl018_fq_kkt_solve_h16(torch::Tensor k, torch::Tensor beta) {
  check_tensor(k, "k");
  check_tensor(beta, "beta");
  TORCH_CHECK(k.dim() == 4, "k must be [B,T,16,128]");
  TORCH_CHECK(beta.dim() == 3, "beta must be [B,T,16]");
  const int batch = static_cast<int>(k.size(0));
  const int tokens = static_cast<int>(k.size(1));
  TORCH_CHECK(k.size(2) == 16 && k.size(3) == 128, "k must be [B,T,16,128]");
  TORCH_CHECK(beta.size(0) == batch && beta.size(1) == tokens && beta.size(2) == 16,
              "beta must be [B,T,16]");
  TORCH_CHECK(tokens % 64 == 0, "tokens must be divisible by 64");

  auto a = torch::empty({batch, tokens, 16, 64}, k.options());
  const int num_chunks = batch * (tokens / 64);

  alignas(64) CUtensorMap a_desc;
  alignas(64) CUtensorMap k_desc;
  check_cuda(cuInit(0), "cuInit");
  make_a_desc(&a_desc, a.data_ptr(), batch, tokens);
  make_k_desc(&k_desc, k.data_ptr(), batch, tokens);

  dim3 grid(num_chunks * 16, 1, 1);
  dim3 block(256, 1, 1);
  constexpr size_t smem = 44160;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  tilelang_kkt_solve_kernel_kernel<<<grid, block, smem, stream>>>(
      reinterpret_cast<half_t*>(a.data_ptr<at::Half>()),
      a_desc,
      reinterpret_cast<const half_t*>(beta.data_ptr<at::Half>()),
      k_desc,
      batch,
      num_chunks,
      tokens);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return a;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("kkt_solve_h16", &tl018_fq_kkt_solve_h16,
        "TL0.1.8 FlashQLA H16 KKT solve launcher");
}
