#include "mul_cuda.h"

#include <iostream>

#include "utils.h"

__device__ inline void swap_kernel(LL &a, LL &b)
{
  LL t = a;
  a = b;
  b = t;
}

__device__ void bit_reverse_device(LL *a, int n)
{
  int j = 0;
  for (int i = 1; i < n; i++)
  {
    int bit = n >> 1;
    while (j & bit)
    {
      j ^= bit;
      bit >>= 1;
    }
    j ^= bit;
    if (i < j)
      swap_kernel(a[i], a[j]);
  }
}

__global__ void bit_reverse_kernel(LL *a, int n)
{
  if (threadIdx.x == 0)
  {
    bit_reverse_device(a, n);
  }
}

__device__ LL REDC_kernel(unsigned __int128 T_128, LL MOD, LL N_inv)
{
  LL MONT_R_mask = R - 1;
  LL m = ((T_128 & MONT_R_mask) * N_inv) & MONT_R_mask;

  unsigned __int128 temp_t = T_128 + (unsigned __int128)m * MOD;
  LL t = temp_t >> MONT_R;
  return t >= MOD ? t - MOD : t;
}

__device__ LL qpow_montgomery_kernel(LL a, LL b, LL p, LL N_inv)
{
  LL a_mont = a * R % p;
  LL ans = 1 * R % p;
  while (b)
  {
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
                                                     unsigned long long b)
{
  return __mul64hi(a, b);
}

// 2. 修正后的Barrett规约核函数
__device__ inline LL barrett_reduce_kernel(LL a, LL p, LL m_inv)
{
  // 调用__mul64hi来获取128位乘积的高64位
  unsigned long long q = get_high_64(a, m_inv);

  // 用高64位的结果来计算余数
  LL t = a - q * p;

  // 做最后的修正
  return t >= p ? t - p : t;
}

__device__ LL qpow_barrett_kernel(LL a, LL b, LL p, LL MOD2)
{
  LL ans = 1;
  while (b)
  {
    if (b & 1)
      ans = barrett_reduce_kernel((1LL * ans * a), p, MOD2);
    a = barrett_reduce_kernel((1LL * a * a), p, MOD2);
    b >>= 1;
  }
  return ans;
}

__device__ LL qpow_kernel(LL a, LL b, LL p)
{
  LL ans = 1;
  while (b)
  {
    if (b & 1)
      ans = (1LL * ans * a) % p;
    a = (1LL * a * a) % p;
    b >>= 1;
  }
  return ans;
}

__global__ void partial_ntt(LL *a, LL *w, int n, int len, LL MOD, LL MOD2)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int i = idx / (len / 2) * len;
  int j = idx % (len / 2);
  if (i + j + len / 2 >= n)
    return;
  LL u = a[i + j], v = 1LL * w[len / 2 - 1 + j] * a[i + j + len / 2] % MOD;
  a[i + j] = (u + v) % MOD;
  a[i + j + len / 2] = (u - v + MOD) % MOD;
}
__global__ void compute_w(LL ROOT, LL MOD, LL MOD2, LL *w, int n, bool invert)
{
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
                                    LL MOD2)
{
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
                                  bool invert)
{
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
                                     bool invert)
{
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

void cuda_ntt(LL *a, int n, LL MOD, LL MOD2, bool invert)
{
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
  for (int len = 2; len <= n; len <<= 1)
  {

    partial_ntt<<<num_blocks, threads_per_block>>>(a_d, w_d, n, len, MOD, MOD2);
    cudaDeviceSynchronize();
  }
  // std::cout<<"结束ntt"<<std::endl;
  cudaMemcpy(a, a_d, sizeof(LL) * n, cudaMemcpyDeviceToHost);
  cudaFree(a_d);
  cudaFree(w_d);
  cudaDeviceSynchronize();

  if (invert)
  {
    LL inv_n = qpow(n, MOD - 2, MOD);
    for (int i = 0; i < n; i++)
      a[i] = 1LL * a[i] * inv_n % MOD;
  }
}

void cuda_ntt_multiply(LL *a, LL *b, LL *ab, int n, LL MOD)
{
  int size = 1;
  //   unsigned __int128 magic_val=1;
  //   magic_val<<=BITE2;
  //   LL MOD2 = magic_val/MOD;
  LL R_inv = inv(MOD, R);
  LL MOD2 = R - R_inv;

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
//------------------------------------------------------------------------------
__device__ LL qpow_kernel_real_montgomery(LL a, LL b, LL p, LL N_inv, LL R2_mod_n)
{
  LL a_mont = REDC_kernel((unsigned __int128)a * R2_mod_n, p, N_inv);
  LL ans = REDC_kernel(1 * R2_mod_n, p, N_inv);
  while (b)
  {
    if (b & 1)
      ans = REDC_kernel((unsigned __int128)ans * a_mont, p, N_inv);
    a_mont = REDC_kernel((unsigned __int128)a_mont * a_mont, p, N_inv);
    b >>= 1;
  }
  return ans;
}

__device__ LL qpow_kernel_real_montgomery2(LL a_mont, LL b, LL p, LL N_inv, LL R2_mod_n)
{
  LL ans = REDC_kernel(1 * R2_mod_n, p, N_inv);
  while (b)
  {
    if (b & 1)
      ans = REDC_kernel((unsigned __int128)ans * a_mont, p, N_inv);
    a_mont = REDC_kernel((unsigned __int128)a_mont * a_mont, p, N_inv);
    b >>= 1;
  }
  return ans;
}
__global__ void assign_ab(LL *a, LL *b, LL *ab, int n, LL MOD, LL N_inv)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
  {
    ab[idx] = REDC_kernel((unsigned __int128)a[idx] * b[idx], MOD, N_inv);
  }
}

__global__ void to_montgomery(LL *a, int n, LL MOD, LL n_inv, LL R2_mod_n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
  {
    a[idx] = REDC_kernel((unsigned __int128)a[idx] * R2_mod_n, MOD, n_inv);
  }
}

__global__ void from_montgomery(LL *a, int n, LL MOD, LL n_inv)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
  {
    a[idx] = REDC_kernel(a[idx], MOD, n_inv);
  }
}

__global__ void assign_a_real_montgomery(LL *a, int n, LL MOD, LL n_inv, LL R2_mod_n)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
  {
    LL inv_n = qpow_montgomery_kernel(n, MOD - 2, MOD, n_inv);
    inv_n = REDC_kernel((unsigned __int128)inv_n * R2_mod_n, MOD, n_inv);
    a[idx] = REDC_kernel((unsigned __int128)a[idx] * inv_n, MOD, n_inv);
  }
}
__global__ void partial_ntt_real_montgomery(LL *a, LL *w, int n, int len, LL MOD, LL MOD2)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int i = idx / (len / 2) * len;
  int j = idx % (len / 2);
  if (i + j + len / 2 >= n)
    return;
  LL u = a[i + j], v = REDC_kernel((unsigned __int128)w[len / 2 - 1 + j] * a[i + j + len / 2], MOD, MOD2);
  a[i + j] = u + v;
  a[i + j + len / 2] = u - v + MOD;
  if (a[i + j] >= MOD)
    a[i + j] -= MOD;
  if (a[i + j + len / 2] >= MOD)
    a[i + j + len / 2] -= MOD;
}

void cuda_ntt_real_montgomery(LL *d_a, LL *d_w, int n, LL MOD, LL MOD2, LL R2_mod_n, bool invert)
{
  int num_blocks = n / 1024 + 1;
  int threads_per_block = 1024;
  for (int len = 2; len <= n; len <<= 1)
  {
    partial_ntt_real_montgomery<<<num_blocks, threads_per_block>>>(d_a, d_w, n, len, MOD, MOD2);
    cudaDeviceSynchronize();
  }
  if (invert)
  {
    assign_a_real_montgomery<<<num_blocks, threads_per_block>>>(d_a, n, MOD, MOD2, R2_mod_n);
    cudaDeviceSynchronize();
  }
}

__global__ void compute_w_real_montgomery(LL ROOT, LL MOD, LL MOD2, LL *d_w, LL R2_mod_n, int n, bool invert)
{
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
  LL wn = qpow_montgomery_kernel(ROOT, (MOD - 1) / len, MOD, MOD2);
  if (invert)
    wn = qpow_montgomery_kernel(wn, MOD - 2, MOD, MOD2);
  wn = REDC_kernel((unsigned __int128)wn * R2_mod_n, MOD, MOD2);

  // 3. 计算 w[j] = (wn)^j，并存入正确的位置 w[idx]
  d_w[idx] = qpow_kernel_real_montgomery2(wn, j, MOD, MOD2, R2_mod_n);
}

void cuda_ntt_multiply_real_montgomery(LL *a, LL *b, LL *ab, int n, LL MOD)
{
  int size = 1;
  while (size < 2 * n)
    size <<= 1;
  for (int i = n; i < size; i++)
    a[i] = b[i] = 0;
  bit_reverse(a, size);
  bit_reverse(b, size);
  
  int num_blocks = size / 1024 + 1;
  int threads_per_block = 1024;

  LL R_inv = inv(MOD, R);
  LL MOD2 = R-R_inv;
  unsigned __int128 magic_val = 1;
  magic_val <<= (MONT_R * 2);
  LL R2_mod_n = magic_val % MOD;
  LL *d_a, *d_b, *d_ab, *d_w;
  cudaMalloc((void **)&d_a, sizeof(LL) * size);
  cudaMalloc((void **)&d_b, sizeof(LL) * size);
  cudaMalloc((void **)&d_ab, sizeof(LL) * size);
  cudaMalloc((void **)&d_w, sizeof(LL) * size);
  cudaMemcpy(d_a, a, sizeof(LL) * size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, sizeof(LL) * size, cudaMemcpyHostToDevice);
  compute_w_real_montgomery<<<num_blocks, threads_per_block>>>(ROOT, MOD, MOD2, d_w, R2_mod_n, size, false);
  to_montgomery<<<num_blocks, threads_per_block>>>(d_a, size, MOD, MOD2, R2_mod_n);
  to_montgomery<<<num_blocks, threads_per_block>>>(d_b, size, MOD, MOD2, R2_mod_n);
  cudaDeviceSynchronize();

  cuda_ntt_real_montgomery(d_a, d_w, size, MOD, MOD2, R2_mod_n, false);
  cuda_ntt_real_montgomery(d_b, d_w, size, MOD, MOD2, R2_mod_n, false);

  assign_ab<<<num_blocks, threads_per_block>>>(d_a, d_b, d_ab, size, MOD, MOD2);
  compute_w_real_montgomery<<<num_blocks, threads_per_block>>>(
      ROOT, MOD, MOD2, d_w, R2_mod_n, size, true);
  cudaDeviceSynchronize();
  cudaMemcpy(ab, d_ab, sizeof(LL) * size, cudaMemcpyDeviceToHost);
  bit_reverse(ab, size);
  cudaMemcpy(d_ab, ab, sizeof(LL) * size, cudaMemcpyHostToDevice);
  cuda_ntt_real_montgomery(d_ab, d_w, size, MOD, MOD2, R2_mod_n, true);
  from_montgomery<<<num_blocks, threads_per_block>>>(d_ab, size, MOD, MOD2);
  cudaDeviceSynchronize();
  cudaMemcpy(ab, d_ab, sizeof(LL) * size, cudaMemcpyDeviceToHost);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_ab);
  cudaFree(d_w);
  cudaDeviceSynchronize();
}