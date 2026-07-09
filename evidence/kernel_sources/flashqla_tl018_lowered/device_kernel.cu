#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void tilelang_kkt_solve_kernel_kernel(half_t* __restrict__ a, __grid_constant__ const CUtensorMap a_desc, const half_t* __restrict__ b, __grid_constant__ const CUtensorMap k_desc, int data_batch_size, int num_chunks, int num_tokens);
extern "C" __global__ void __launch_bounds__(256, 1) tilelang_kkt_solve_kernel_kernel(half_t* __restrict__ a, __grid_constant__ const CUtensorMap a_desc, const half_t* __restrict__ b, __grid_constant__ const CUtensorMap k_desc, int data_batch_size, int num_chunks, int num_tokens) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  __shared__ __align__(16) uint64_t k_is_ready_mem[1];
  auto k_is_ready = reinterpret_cast<Barrier*>(k_is_ready_mem);
  __shared__ __align__(16) uint64_t a_is_ready_mem[1];
  auto a_is_ready = reinterpret_cast<Barrier*>(a_is_ready_mem);
  int batch_idx = 0;
  int chunk_idx = 0;
  int seq_start_idx = 0;
  int seq_end_idx = 0;
  __shared__ __align__(16) float b_shared[64];
  float a64_fragment[32];
  float a16i_row[16];
  float a16i_sum[1];
  float a16o_fragment[4];
  float a32i_fragment[16];
  float a32o_fragment[8];
  half_t a64_shared_local_cast[4];
  half_t a64_shared_local_cast_1[2];
  if (tl::tl_shuffle_elect<0>()) {
    tl::prefetch_tma_descriptor(k_desc);
    tl::prefetch_tma_descriptor(a_desc);
  }
  if (tl::tl_shuffle_elect<0>()) {
    k_is_ready[0].init(32);
    a_is_ready[0].init(128);
  }
  tl::fence_barrier_init();
  __syncthreads();
  batch_idx = ((((int)blockIdx.x) >> 4) % data_batch_size);
  chunk_idx = ((((int)blockIdx.x) >> 4) / data_batch_size);
  seq_start_idx = 0;
  seq_end_idx = num_tokens;
  int chunk_idx_1 = chunk_idx;
  int seq_start_idx_1 = seq_start_idx;
  int seq_end_idx_1 = seq_end_idx;
  if (((int)threadIdx.x) < 128) {
    tl::warpgroup_reg_alloc<64>();
    if ((((chunk_idx_1 * 64) + seq_start_idx_1) + 64) <= seq_end_idx_1) {
      if (((int)threadIdx.x) < 64) {
        half_t condval;
        if (((0 <= (((chunk_idx_1 * 64) + seq_start_idx_1) + ((int)threadIdx.x))) && ((((chunk_idx_1 * 64) + seq_start_idx_1) + ((int)threadIdx.x)) < num_tokens))) {
          condval = b[(((((((int64_t)chunk_idx_1) * (int64_t)1024) + (((int64_t)seq_start_idx_1) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) * (int64_t)16)) + ((((((int64_t)((int)blockIdx.x)) >> (int64_t)4) % ((int64_t)data_batch_size)) * ((int64_t)num_tokens)) * (int64_t)16)) + (((int64_t)((int)blockIdx.x)) & (int64_t)15))];
        } else {
          condval = half_t(0x0p+0f/*0.000000e+00*/);
        }
        b_shared[((int)threadIdx.x)] = ((float)condval);
      }
    } else {
      if (((int)threadIdx.x) < 64) {
        if ((((chunk_idx_1 * 64) + seq_start_idx_1) + ((int)threadIdx.x)) < seq_end_idx_1) {
          half_t condval_1;
          if (((0 <= (((chunk_idx_1 * 64) + seq_start_idx_1) + ((int)threadIdx.x))) && ((((chunk_idx_1 * 64) + seq_start_idx_1) + ((int)threadIdx.x)) < num_tokens))) {
            condval_1 = b[(((((((int64_t)chunk_idx_1) * (int64_t)1024) + (((int64_t)seq_start_idx_1) * (int64_t)16)) + (((int64_t)((int)threadIdx.x)) * (int64_t)16)) + ((((((int64_t)((int)blockIdx.x)) >> (int64_t)4) % ((int64_t)data_batch_size)) * ((int64_t)num_tokens)) * (int64_t)16)) + (((int64_t)((int)blockIdx.x)) & (int64_t)15))];
          } else {
            condval_1 = half_t(0x0p+0f/*0.000000e+00*/);
          }
          b_shared[((int)threadIdx.x)] = ((float)condval_1);
        } else {
          b_shared[((int)threadIdx.x)] = 0x0p+0f/*0.000000e+00*/;
        }
      }
    }
    k_is_ready[0].wait(0);
    tl::fence_proxy_async();
    tl::gemm_ss<64, 64, 128, 4, 1, 0, 1, 1, 128, 128, 0, 0, true>((&(((half_t*)buf_dyn_shmem)[0])), (&(((half_t*)buf_dyn_shmem)[0])), (&(a64_fragment[0])));
    tl::__sync_thread_partial<3, 128>();
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      float2 __1;
        float2 v_ = *(float2*)(a64_fragment + (i * 2));
        float2 v__1 = make_float2(b_shared[((((((int)threadIdx.x) >> 5) * 16) + ((i & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], b_shared[((((((int)threadIdx.x) >> 5) * 16) + ((i & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
        __1.x = (v_.x*v__1.x);
        __1.y = (v_.y*v__1.y);
      *(float2*)(a64_fragment + (i * 2)) = __1;
    }
    #pragma unroll
    for (int i_1 = 0; i_1 < 32; ++i_1) {
      if (((((((int)threadIdx.x) >> 5) * 16) + (((i_1 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < ((((i_1 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_1 & 1))) {
        a64_fragment[i_1] = 0x0p+0f/*0.000000e+00*/;
      } else {
        if (((((((int)threadIdx.x) >> 5) * 16) + (((i_1 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) == ((((i_1 >> 2) * 8) + ((((int)threadIdx.x) & 3) * 2)) + (i_1 & 1))) {
          a64_fragment[i_1] = 0x1p+0f/*1.000000e+00*/;
        }
      }
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 16; ++i_2) {
      if ((64 <= ((int)threadIdx.x)) && (i_2 < 8)) {
        float broadcast_var = -0x1p+0f/*-1.000000e+00*/;
        float2 __2;
          float2 v__2 = *(float2*)(a64_fragment + (i_2 * 2));
          float2 v__3 = make_float2(broadcast_var, broadcast_var);
          __2.x = (v__2.x*v__3.x);
          __2.y = (v__2.y*v__3.y);
        *(float2*)(((float*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) >> 5) * 512) + ((i_2 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_2 >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((i_2 & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 8448)) = __2;
      } else {
        if ((((int)threadIdx.x) >> 5) == ((i_2 >> 2) + 1)) {
          float broadcast_var_1 = -0x1p+0f/*-1.000000e+00*/;
          float2 __3;
            float2 v__4 = *(float2*)(a64_fragment + (i_2 * 2));
            float2 v__5 = make_float2(broadcast_var_1, broadcast_var_1);
            __3.x = (v__4.x*v__5.x);
            __3.y = (v__4.y*v__5.y);
          *(float2*)(((float*)buf_dyn_shmem) + (((((((((int)threadIdx.x) >> 6) * 272) + ((i_2 & 1) * 128)) + (((((int)threadIdx.x) & 31) >> 2) * 16)) + (((i_2 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 10496)) = __3;
        } else {
          if ((((int)threadIdx.x) >> 5) == (i_2 >> 2)) {
            *(float2*)(((float*)buf_dyn_shmem) + (((((((((int)threadIdx.x) >> 5) * 272) + ((i_2 & 1) * 128)) + (((((int)threadIdx.x) & 31) >> 2) * 16)) + (((i_2 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 6144)) = *(float2*)(a64_fragment + (i_2 * 2));
          }
        }
      }
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 4; ++i_3) {
      float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(a16i_row + (i_3 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
    }
    tl::__sync_thread_partial<3, 128>();
    #pragma unroll
    for (int k_s = 1; k_s < 16; ++k_s) {
      #pragma unroll
      for (int i_4 = 0; i_4 < 4; ++i_4) {
        for (int vec_s = 0; vec_s < 4; ++vec_s) {
          if (((i_4 * 4) + vec_s) < k_s) {
            a16i_row[((i_4 * 4) + vec_s)] = ((float*)buf_dyn_shmem)[(((((((((int)threadIdx.x) & 63) >> 4) * 272) + (k_s * 16)) + (i_4 * 4)) + vec_s) + 6144)];
          }
        }
      }
      a16i_sum[0] = 0x0p+0f/*0.000000e+00*/;
      #pragma unroll
      for (int k_r = 0; k_r < k_s; ++k_r) {
        a16i_sum[0] = (a16i_sum[0] - (((float*)buf_dyn_shmem)[((((((((int)threadIdx.x) & 63) >> 4) * 272) + (k_r * 16)) + (((int)threadIdx.x) & 15)) + 6144)] * a16i_row[k_r]));
      }
      tl::__sync_thread_partial<3, 128>();
      if ((((int)threadIdx.x) >> 6) == 0) {
        if ((((int)threadIdx.x) & 15) < k_s) {
          ((float*)buf_dyn_shmem)[((((((((int)threadIdx.x) & 63) >> 4) * 272) + (k_s * 16)) + (((int)threadIdx.x) & 15)) + 6144)] = a16i_sum[0];
        }
      }
    }
    float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(a16o_fragment + 0) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
    tl::__sync_thread_partial<3, 128>();
    #pragma unroll
    for (int k_r_1 = 0; k_r_1 < 16; ++k_r_1) {
      float4 __4;
        float4 v__6 = *(float4*)(a16o_fragment + 0);
        float4 __5;
          float4 v__7 = make_float4(((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 6) * 544) + (((((int)threadIdx.x) & 63) >> 2) * 16)) + k_r_1) + 6416)], ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 6) * 544) + (((((int)threadIdx.x) & 63) >> 2) * 16)) + k_r_1) + 6416)], ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 6) * 544) + (((((int)threadIdx.x) & 63) >> 2) * 16)) + k_r_1) + 6416)], ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 6) * 544) + (((((int)threadIdx.x) & 63) >> 2) * 16)) + k_r_1) + 6416)]);
          float4 v__8 = *(float4*)(((float*)buf_dyn_shmem) + (((((((int)threadIdx.x) >> 6) * 272) + (k_r_1 * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 10496));
          __5.x = (v__7.x*v__8.x);
          __5.y = (v__7.y*v__8.y);
          __5.z = (v__7.z*v__8.z);
          __5.w = (v__7.w*v__8.w);
        __4.x = (v__6.x+__5.x);
        __4.y = (v__6.y+__5.y);
        __4.z = (v__6.z+__5.z);
        __4.w = (v__6.w+__5.w);
      *(float4*)(a16o_fragment + 0) = __4;
    }
    tl::__sync_thread_partial<3, 128>();
    #pragma unroll
    for (int i_5 = 0; i_5 < 4; ++i_5) {
      ((float*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 6) * 272) + ((((int)threadIdx.x) & 3) * 64)) + (i_5 * 16)) + ((((int)threadIdx.x) & 63) >> 2)) + 10496)] = a16o_fragment[i_5];
    }
    float broadcast_var_4 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(a16o_fragment + 0) = make_float4(broadcast_var_4, broadcast_var_4, broadcast_var_4, broadcast_var_4);
    tl::__sync_thread_partial<3, 128>();
    #pragma unroll
    for (int k_r_2 = 0; k_r_2 < 16; ++k_r_2) {
      float4 __6;
        float4 v__9 = *(float4*)(a16o_fragment + 0);
        float4 __7;
          float4 v__10 = make_float4(((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 6) * 272) + (k_r_2 * 16)) + ((((int)threadIdx.x) & 63) >> 2)) + 10496)], ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 6) * 272) + (k_r_2 * 16)) + ((((int)threadIdx.x) & 63) >> 2)) + 10496)], ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 6) * 272) + (k_r_2 * 16)) + ((((int)threadIdx.x) & 63) >> 2)) + 10496)], ((float*)buf_dyn_shmem)[(((((((int)threadIdx.x) >> 6) * 272) + (k_r_2 * 16)) + ((((int)threadIdx.x) & 63) >> 2)) + 10496)]);
          float4 v__11 = *(float4*)(((float*)buf_dyn_shmem) + (((((((int)threadIdx.x) >> 6) * 544) + (k_r_2 * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 6144));
          __7.x = (v__10.x*v__11.x);
          __7.y = (v__10.y*v__11.y);
          __7.z = (v__10.z*v__11.z);
          __7.w = (v__10.w*v__11.w);
        __6.x = (v__9.x+__7.x);
        __6.y = (v__9.y+__7.y);
        __6.z = (v__9.z+__7.z);
        __6.w = (v__9.w+__7.w);
      *(float4*)(a16o_fragment + 0) = __6;
    }
    tl::__sync_thread_partial<3, 128>();
    *(float4*)(((float*)buf_dyn_shmem) + ((((((int)threadIdx.x) >> 6) * 272) + ((((int)threadIdx.x) & 63) * 4)) + 10496)) = *(float4*)(a16o_fragment + 0);
    #pragma unroll
    for (int i_6 = 0; i_6 < 16; ++i_6) {
      if (((i_6 & 7) < 4) && (4 <= (((int)threadIdx.x) & 7))) {
        a32i_fragment[i_6] = 0x0p+0f/*0.000000e+00*/;
      }
    }
    tl::__sync_thread_partial<3, 128>();
    #pragma unroll
    for (int i_7 = 0; i_7 < 4; ++i_7) {
      if (((i_7 & 1) == 1) && ((((int)threadIdx.x) & 7) < 4)) {
        *(float4*)(a32i_fragment + (i_7 * 4)) = *(float4*)(((float*)buf_dyn_shmem) + ((((((i_7 >> 1) * 272) + ((i_7 & 1) * 256)) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)threadIdx.x) & 7) * 4)) + 10240));
      }
    }
    #pragma unroll
    for (int i_8 = 0; i_8 < 4; ++i_8) {
      if ((i_8 & 1) == ((((int)threadIdx.x) & 7) >> 2)) {
        *(float4*)(a32i_fragment + (i_8 * 4)) = *(float4*)(((float*)buf_dyn_shmem) + ((((i_8 * 272) + ((((int)threadIdx.x) >> 3) * 16)) + ((((int)threadIdx.x) & 3) * 4)) + 6144));
      }
    }
    #pragma unroll
    for (int i_9 = 0; i_9 < 4; ++i_9) {
      if ((i_9 >> 1) == 0) {
        *(float4*)(((float*)buf_dyn_shmem) + (((((((i_9 & 1) * 512) + ((((int)threadIdx.x) >> 3) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 4)) + 7424)) = *(float4*)(a32i_fragment + (i_9 * 4));
      } else {
        *(float4*)(((float*)buf_dyn_shmem) + (((((((i_9 & 1) * 512) + ((((int)threadIdx.x) >> 3) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 4)) + 8448)) = *(float4*)(a32i_fragment + (i_9 * 4));
      }
    }
    tl::fence_proxy_async();
    tl::__sync_thread_partial<3, 128>();
    tl::gemm_ss<32, 32, 32, 2, 2, 0, 0, 1, 32, 32, 0, 0, false>((&(((float*)buf_dyn_shmem)[8448])), (&(((float*)buf_dyn_shmem)[9472])), (&(a32o_fragment[0])));
    tl::__sync_thread_partial<3, 128>();
    #pragma unroll
    for (int i_10 = 0; i_10 < 4; ++i_10) {
      *(float2*)(((float*)buf_dyn_shmem) + ((((((((((((int)threadIdx.x) & 63) >> 5) * 512) + ((i_10 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 2) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (i_10 >> 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 8)) + (((((((int)threadIdx.x) & 7) >> 2) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + 9472)) = *(float2*)(a32o_fragment + (i_10 * 2));
    }
    tl::fence_proxy_async();
    tl::__sync_thread_partial<3, 128>();
    tl::gemm_ss<32, 32, 32, 2, 2, 0, 0, 1, 32, 32, 0, 0, false>((&(((float*)buf_dyn_shmem)[9472])), (&(((float*)buf_dyn_shmem)[7424])), (&(a32o_fragment[0])));
    tl::__sync_thread_partial<3, 128>();
    #pragma unroll
    for (int i_11 = 0; i_11 < 4; ++i_11) {
      uint2 __8;
      float4 v__12 = *(float4*)(a32i_fragment + (i_11 * 4));
      ((half2*)(&__8))[0] = __float22half2_rn(((float2*)(&v__12))[0]);
      ((half2*)(&__8))[1] = __float22half2_rn(((float2*)(&v__12))[1]);
      *(uint2*)(a64_shared_local_cast + 0) = __8;
      *(uint2*)(((half_t*)buf_dyn_shmem) + (((((((i_11 * 1024) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + (i_11 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 8192)) = *(uint2*)(a64_shared_local_cast + 0);
    }
    tl::__sync_thread_partial<3, 128>();
    #pragma unroll
    for (int i_12 = 0; i_12 < 4; ++i_12) {
      uint1 __9;
      float2 v__13 = *(float2*)(a32o_fragment + (i_12 * 2));
      ((half2*)(&__9))[0] = __float22half2_rn(((float2*)(&v__13))[0]);
      *(uint1*)(a64_shared_local_cast_1 + 0) = __9;
      *(uint1*)(((half_t*)buf_dyn_shmem) + ((((((((((((int)threadIdx.x) & 63) >> 5) * 1024) + ((i_12 & 1) * 512)) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((int)threadIdx.x) & 31) >> 4) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (i_12 >> 1)) & 1) * 16)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 10240)) = *(uint1*)(a64_shared_local_cast_1 + 0);
    }
    half_t broadcast_var_5 = half_t(0x0p+0f/*0.000000e+00*/);
    *(uint4*)(((half_t*)buf_dyn_shmem) + ((((((((int)threadIdx.x) >> 2) * 64) + (((((((int)threadIdx.x) & 31) >> 4) + 1) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 8192)) = make_uint4(__pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5));
    a_is_ready[0].arrive();
  } else {
    tl::warpgroup_reg_dealloc<24>();
    if (((int)threadIdx.x) < 160) {
      if (((int)threadIdx.x) == 128) {
        k_is_ready[0].expect_transaction(16384);
        tl::fence_proxy_async();
        tl::tma_load(k_desc, k_is_ready[0], (&(((half_t*)buf_dyn_shmem)[0])), 0, (((int)blockIdx.x) & 15), ((chunk_idx_1 * 64) + seq_start_idx_1), ((((int)blockIdx.x) >> 4) % data_batch_size));
        tl::tma_load(k_desc, k_is_ready[0], (&(((half_t*)buf_dyn_shmem)[4096])), 64, (((int)blockIdx.x) & 15), ((chunk_idx_1 * 64) + seq_start_idx_1), ((((int)blockIdx.x) >> 4) % data_batch_size));
      }
      k_is_ready[0].arrive();
    } else {
      if (((int)threadIdx.x) < 192) {
        a_is_ready[0].wait(0);
        tl::fence_proxy_async();
        if ((((chunk_idx_1 * 64) + seq_start_idx_1) + 64) <= seq_end_idx_1) {
          if (((int)threadIdx.x) == 160) {
            tl::tma_store(a_desc, (&(((half_t*)buf_dyn_shmem)[8192])), 0, (((int)blockIdx.x) & 15), ((chunk_idx_1 * 64) + seq_start_idx_1), ((((int)blockIdx.x) >> 4) % data_batch_size));
            tl::tma_store_arrive();
            tl::tma_store_wait<0>();
          }
        }
      } else {
        a_is_ready[0].wait(0);
        if (seq_end_idx_1 < (((chunk_idx_1 * 64) + seq_start_idx_1) + 64)) {
          #pragma unroll
          for (int i_13 = 0; i_13 < 8; ++i_13) {
            if (((((chunk_idx_1 * 64) + (i_13 * 8)) + (((int)threadIdx.x) >> 3)) + seq_start_idx_1) < (seq_end_idx_1 + 24)) {
              if (24 <= ((((chunk_idx_1 * 64) + (i_13 * 8)) + (((int)threadIdx.x) >> 3)) + seq_start_idx_1)) {
                if (((((chunk_idx_1 * 64) + (i_13 * 8)) + (((int)threadIdx.x) >> 3)) + seq_start_idx_1) < (num_tokens + 24)) {
                  *(uint4*)(a + ((((((((((int64_t)chunk_idx_1) * (int64_t)65536) + (((int64_t)i_13) * (int64_t)8192)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)1024)) + (((int64_t)seq_start_idx_1) * (int64_t)1024)) + ((((((int64_t)((int)blockIdx.x)) >> (int64_t)4) % ((int64_t)data_batch_size)) * ((int64_t)num_tokens)) * (int64_t)1024)) + ((((int64_t)((int)blockIdx.x)) & (int64_t)15) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)) - (int64_t)24576)) = *(uint4*)(((half_t*)buf_dyn_shmem) + ((((((i_13 * 512) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 6656));
                }
              }
            }
          }
        }
      }
    }
  }
}

