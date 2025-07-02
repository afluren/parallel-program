// #include "mul_cuda.h"

// #include <iostream>

// #include "utils.h"

// __global__ void partial_ntt(LL *a, LL*w, int n, int len, LL MOD){
//     int block_i = blockIdx.x;
//     int thread_i = threadIdx.x;
//     int idx = block_i * blockDim.x + thread_i;
//     int i = idx / len*len;
//     int j = idx % len;
//     if(i+j+len/2 >= n  || j >= len/2)return;
//     LL u = a[i + j], v = 1LL * w[j] * a[i + j + len / 2] % MOD;
//     a[i + j] = (u + v) % MOD;
//     a[i + j + len / 2] = (u - v+MOD) % MOD;
// }

// __device__ LL qpow_kernel(LL a, LL b, LL p) {
//     LL ans = 1;
//     while (b) {
//         if (b & 1) ans = (1LL*ans * a) % p;
//         a = (1LL*a * a) % p;
//         b >>= 1;
//     }
//     return ans;
// }

// void cuda_ntt(LL *a,int n,LL MOD,bool invert) {
//     bit_reverse(a, n);
//     LL *a_d;
//     cudaMalloc((void**)&a_d, sizeof(LL) * n);
//     cudaMemcpy(a_d, a, sizeof(LL) * n, cudaMemcpyHostToDevice);

//     for (int len = 2; len <= n; len <<= 1) {
//         LL wn = qpow(ROOT, (MOD - 1) / len, MOD);
//         if (invert) wn = qpow(wn, MOD - 2, MOD);
//         LL* w = new LL[len / 2];
//         w[0] = 1;
//         for (int i = 1; i < len / 2; i++) w[i] = 1LL * wn * w[i - 1] % MOD;

//         LL* w_d;
//         cudaMalloc((void**)&w_d, sizeof(LL) * (len / 2));
//         cudaMemcpy(w_d, w, sizeof(LL) * (len / 2), cudaMemcpyHostToDevice);
//         int num_blocks = n/1024+1;
//         int threads_per_block = 1024;
//         partial_ntt<<<num_blocks, threads_per_block>>>(a_d, w_d, n,len, MOD);
//         cudaDeviceSynchronize();
// std::cout<<"len = "<<len<<std::endl;
//         delete[] w;
//         cudaFree(w_d);
// std::cout<<"结束一次"<<std::endl;
//     }
// std::cout<<"开始完全"<<std::endl;
//     cudaMemcpy(a, a_d, sizeof(LL) * n, cudaMemcpyDeviceToHost);
//     cudaFree(a_d);
// std::cout<<"完全结束"<<std::endl;

//     if (invert) {
//         LL inv_n = qpow(n, MOD - 2, MOD);
//         for (int i = 0; i < n; i++) a[i] = 1LL * a[i] * inv_n % MOD;
//     }
// }

// // #include "mul_cuda.h"
// // #include "utils.h"

// // // __global__ void partial_ntt(...)  -> 我们将修改这个kernel
// // __global__ void partial_ntt(LL *a, LL*w, int n, int len, LL MOD){
// //     int block_i = blockIdx.x;
// //     int thread_j_start = threadIdx.x; // 每个线程的起始局部索引

// //     // 计算这个block负责处理的数据段的基地址
// //     int group_base_idx = block_i * len;

// //     // 使用固定的线程数，通过循环处理一个group内的所有蝶形运算
// //     // 每个线程的步长是 blockDim.x (block内的线程总数)
// //     for (int j = thread_j_start; j < len / 2; j += blockDim.x) {

// //         int current_idx = group_base_idx + j;

// //         // 读取操作数
// //         LL u = a[current_idx];
// //         // 从w数组中读取对应的旋转因子，并计算v
// //         LL v = (1LL * w[j] * a[current_idx + len/2]) % MOD;

// //         // 执行蝶形运算并写回
// //         a[current_idx] = (u + v) % MOD;
// //         a[current_idx + len/2] = (u - v + MOD) % MOD;
// //     }
// // }

// // void cuda_ntt(LL *a,int n,LL MOD,bool invert) {
// //     bit_reverse(a, n);
// //     LL *a_d;
// //     cudaMalloc((void**)&a_d, sizeof(LL) * n);
// //     cudaMemcpy(a_d, a, sizeof(LL) * n, cudaMemcpyHostToDevice);

// //     for (int len = 2; len <= n; len <<= 1) {
// //         LL wn = qpow(ROOT, (MOD - 1) / len, MOD);
// //         if (invert) wn = qpow(wn, MOD - 2, MOD);

// //         // 注意，这里的w数组大小是 len/2
// //         LL* w = new LL[len / 2];
// //         w[0] = 1;
// //         for (int i = 1; i < len / 2; i++) w[i] = 1LL * wn * w[i - 1] %
// MOD;

// //         LL* w_d;
// //         cudaMalloc((void**)&w_d, sizeof(LL) * (len / 2));
// //         cudaMemcpy(w_d, w, sizeof(LL) * (len / 2),
// cudaMemcpyHostToDevice);

// //         // --- 这是修改的关键部分 ---
// //         int num_blocks = n / len;
// //         // 使用一个固定的、小于1024的线程数
// //         int threads_per_block = 512;

// //         // 当需要处理的任务数(len/2)小于我们设定的线程数时，
// //         // 没必要启动那么多线程，动态调整以节省资源
// //         if (len / 2 < threads_per_block) {
// //             threads_per_block = len / 2;
// //         }

// //         // 确保线程数至少为1 (在 len=2 的情况下, len/2=1)
// //         if (threads_per_block == 0) threads_per_block = 1;

// //         partial_ntt<<<num_blocks, threads_per_block>>>(a_d, w_d, n, len,
// MOD);
// //         // --- 修改结束 ---

// //         cudaDeviceSynchronize(); // 等待kernel执行完成
// //         delete[] w;
// //         cudaFree(w_d);
// //     }
// //     cudaMemcpy(a, a_d, sizeof(LL) * n, cudaMemcpyDeviceToHost);
// //     cudaFree(a_d);

// //     if (invert) {
// //         LL inv_n = qpow(n, MOD - 2, MOD);
// //         for (int i = 0; i < n; i++) a[i] = 1LL * a[i] * inv_n % MOD;
// //     }
// // }
// void cuda_ntt_multiply(LL *a, LL *b, LL *ab, int n,LL MOD) {
//     int size=1;
//     while(size<2*n) size<<=1;
//     for(int i=n;i<size;i++) a[i]=b[i]=0;
//     cuda_ntt(a, size,MOD,false);
//     cuda_ntt(b,size,MOD, false);
//     for (int i = 0; i < size; i++) ab[i] = 1LL * a[i] * b[i] % MOD;
//     cuda_ntt(ab,size,MOD, true);
// }

#include "mul_cuda.h"

#include <iostream>

#include "utils.h"

__device__ LL REDC_kernel(LL T, LL MOD, LL N_inv) {
  unsigned __int128 T_128 = T;
  LL MONT_R_mask = (1LL << MONT_R) - 1;
  LL m = ((T_128 & MONT_R_mask) * N_inv) & MONT_R_mask;

  unsigned __int128 temp_t = T_128 + (unsigned __int128)m * MOD;
  LL t = temp_t >> MONT_R;
  return t >= MOD ? t - MOD : t;
}

__device__ LL qpow_montgomery_kernel(LL a, LL b, LL p, LL N_inv) {
  LL a_mont = a * (1LL << MONT_R) % p;
  LL ans = 1 * (1LL << MONT_R) % p;
  while (b) {
    if (b & 1)
      ans = REDC_kernel(ans * a_mont, p, N_inv);
    a_mont = REDC_kernel(a_mont * a_mont, p, N_inv);
    b >>= 1;
  }
  return REDC_kernel(ans, p, N_inv);
}

// 1. 使用内联汇编或__mul64hi实现精确的128位乘法，得到高64位
//    这里提供一个更通用的版本
__device__ __inline__ unsigned long long get_high_64(unsigned long long a,
                                                     unsigned long long b) {
  return __mul64hi(a, b);
}

// 2. 修正后的Barrett规约核函数
__device__ inline LL barrett_reduce_kernel(LL a, LL p, LL m_inv) {
  // 调用__mul64hi来获取128位乘积的高64位
  unsigned long long q = get_high_64(a, m_inv);

  // 用高64位的结果来计算余数
  LL t = a - q * p;

  // 做最后的修正
  return t >= p ? t - p : t;
}

__device__ LL qpow_barrett_kernel(LL a, LL b, LL p, LL MOD2) {
  LL ans = 1;
  while (b) {
    if (b & 1)
      ans = barrett_reduce_kernel((1LL * ans * a), p, MOD2);
    a = barrett_reduce_kernel((1LL * a * a), p, MOD2);
    b >>= 1;
  }
  return ans;
}

__device__ LL qpow_kernel(LL a, LL b, LL p) {
  LL ans = 1;
  while (b) {
    if (b & 1)
      ans = (1LL * ans * a) % p;
    a = (1LL * a * a) % p;
    b >>= 1;
  }
  return ans;
}

__global__ void partial_ntt(LL *a, LL *w, int n, int len, LL MOD, LL MOD2) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int i = idx / (len / 2) * len;
  int j = idx % (len / 2);
  if (i + j + len / 2 >= n)
    return;
  LL u = a[i + j], v = 1LL * w[len / 2 - 1 + j] * a[i + j + len / 2] % MOD;
  a[i + j] = (u + v) % MOD;
  a[i + j + len / 2] = (u - v + MOD) % MOD;
}
__global__ void compute_w(LL ROOT, LL MOD, LL MOD2, LL *w, int n, bool invert) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // 我们总共需要计算 n-1 个旋转因子
  if (idx >= n - 1)
    return;

  // 1. 根据全局索引idx，反推出它属于哪一层(len)以及是这一层的第几个(j)
  //    这个推导是 w[(len/2 - 1) + j] = w[idx] 的逆运算
  LL val = idx + 1; // 我们需要找大于idx+1的最小2的幂，等价于找idx+2的“上取整”幂
  int bit_width = 64 - __clzll(val);
  int len = 1LL << bit_width;
  int j = idx - ((len >> 1) - 1);

  // 2. 为推导出的len，计算它的主n次单位根 wn
  LL wn = qpow_kernel(ROOT, (MOD - 1) / len, MOD);
  if (invert)
    wn = qpow_kernel(wn, MOD - 2, MOD);

  // 3. 计算 w[j] = (wn)^j，并存入正确的位置 w[idx]
  w[idx] = qpow_kernel(wn, j, MOD);
}

__global__ void partial_ntt_barrett(LL *a, LL *w, int n, int len, LL MOD,
                                    LL MOD2) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int i = idx / (len / 2) * len;
  int j = idx % (len / 2);
  if (i + j + len / 2 >= n)
    return;
  LL u = a[i + j],
     v = barrett_reduce_kernel(1LL * w[len / 2 - 1 + j] * a[i + j + len / 2],
                               MOD, MOD2);
  a[i + j] = barrett_reduce_kernel((u + v), MOD, MOD2);
  a[i + j + len / 2] = barrett_reduce_kernel((u - v + MOD), MOD, MOD2);
}

__global__ void compute_w_barrett(LL ROOT, LL MOD, LL MOD2, LL *w, int n,
                                  bool invert) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // 我们总共需要计算 n-1 个旋转因子
  if (idx >= n - 1)
    return;

  // 1. 根据全局索引idx，反推出它属于哪一层(len)以及是这一层的第几个(j)
  //    这个推导是 w[(len/2 - 1) + j] = w[idx] 的逆运算
  long long val =
      idx + 2; // 我们需要找大于idx+1的最小2的幂，等价于找idx+2的“上取整”幂
  int bit_width = 64 - __clzll(val - 1);
  int len = 1LL << bit_width;
  int j = idx - ((len >> 1) - 1);

  // 2. 为推导出的len，计算它的主n次单位根 wn
  LL wn = qpow_barrett_kernel(ROOT, (MOD - 1) / len, MOD, MOD2);
  if (invert)
    wn = qpow_barrett_kernel(wn, MOD - 2, MOD, MOD2);

  // 3. 计算 w[j] = (wn)^j，并存入正确的位置 w[idx]
  w[idx] = qpow_barrett_kernel(wn, j, MOD, MOD2);
}

__global__ void compute_w_montgomery(LL ROOT, LL MOD, LL MOD2, LL *w, int n,
                                     bool invert) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // 我们总共需要计算 n-1 个旋转因子
  if (idx >= n - 1)
    return;

  // 1. 根据全局索引idx，反推出它属于哪一层(len)以及是这一层的第几个(j)
  //    这个推导是 w[(len/2 - 1) + j] = w[idx] 的逆运算
  long long val =
      idx + 2; // 我们需要找大于idx+1的最小2的幂，等价于找idx+2的“上取整”幂
  int bit_width = 64 - __clzll(val - 1);
  int len = 1LL << bit_width;
  int j = idx - ((len >> 1) - 1);

  // 2. 为推导出的len，计算它的主n次单位根 wn
  LL wn = qpow_montgomery_kernel(ROOT, (MOD - 1) / len, MOD, MOD2);
  if (invert)
    wn = qpow_montgomery_kernel(wn, MOD - 2, MOD, MOD2);

  // 3. 计算 w[j] = (wn)^j，并存入正确的位置 w[idx]
  w[idx] = qpow_montgomery_kernel(wn, j, MOD, MOD2);
}

void cuda_ntt(LL *a, int n, LL MOD, LL MOD2, bool invert) {
  bit_reverse(a, n);
  // LL * w = new LL[n];
  // std::fill(w, w + n, 0);
  LL *a_d;
  LL *w_d;
  // LL *w = new LL[n];
  // std::cout<<"开始分配"<<std::endl;
  cudaMalloc((void **)&a_d, sizeof(LL) * n);
  cudaMalloc((void **)&w_d, sizeof(LL) * n);
  cudaMemcpy(a_d, a, sizeof(LL) * n, cudaMemcpyHostToDevice);
  // cudaMemcpy(w_d, w, sizeof(LL) * n, cudaMemcpyHostToDevice);
  int num_blocks = n / 1024 + 1;
  int threads_per_block = 1024;
  // std::cout<<"开始计算w"<<std::endl;
  compute_w<<<num_blocks, threads_per_block>>>(ROOT, MOD, MOD2, w_d,
                                                          n, invert);
  cudaDeviceSynchronize();
  // std::cout<<"开始ntt"<<std::endl;
  for (int len = 2; len <= n; len <<= 1) {

    partial_ntt<<<num_blocks, threads_per_block>>>(a_d, w_d, n, len, MOD, MOD2);
    cudaDeviceSynchronize();
  }
  // std::cout<<"结束ntt"<<std::endl;
  cudaMemcpy(a, a_d, sizeof(LL) * n, cudaMemcpyDeviceToHost);
  cudaFree(a_d);
  cudaFree(w_d);
  cudaDeviceSynchronize();

  if (invert) {
    LL inv_n = qpow(n, MOD - 2, MOD);
    for (int i = 0; i < n; i++)
      a[i] = 1LL * a[i] * inv_n % MOD;
  }
}

void cuda_ntt_multiply(LL *a, LL *b, LL *ab, int n, LL MOD) {
  int size = 1;
//   unsigned __int128 magic_val=1;
//   magic_val<<=BITE2;
//   LL MOD2 = magic_val/MOD;
  LL R_inv = inv(MOD, 1LL << MONT_R);
  LL MOD2 = (1LL << MONT_R) - R_inv;

  while (size < 2 * n)
    size <<= 1;
  for (int i = n; i < size; i++)
    a[i] = b[i] = 0;
  cuda_ntt(a, size, MOD, MOD2, false);
  cuda_ntt(b, size, MOD, MOD2, false);
  for (int i = 0; i < size; i++)
    ab[i] = 1LL * a[i] * b[i] % MOD;
  cuda_ntt(ab, size, MOD, MOD2, true);
}
