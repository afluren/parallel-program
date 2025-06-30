#include "mul_cuda.h"

#include <iostream>

#include "utils.h"


__global__ void partial_ntt(LL *a, LL*w, int n, int len, LL MOD){
    int block_i = blockIdx.x;
    int thread_i = threadIdx.x;
    int idx = block_i * blockDim.x + thread_i;
    int i = idx / len*len;
    int j = idx % len;
    if(i+j+len/2 >= n  || j >= len/2)return;
    LL u = a[i + j], v = 1LL * w[j] * a[i + j + len / 2] % MOD;
    a[i + j] = (u + v) % MOD;
    a[i + j + len / 2] = (u - v+MOD) % MOD;
}

__device__ LL qpow_kernel(LL a, LL b, LL p) {
    LL ans = 1;
    while (b) {
        if (b & 1) ans = (1LL*ans * a) % p;
        a = (1LL*a * a) % p;
        b >>= 1;
    }
    return ans;
}

__global__ void compute_w(LL ROOT, LL MOD, LL* w, LL n, bool invert){
    int block_i = blockIdx.x;
    int thread_i = threadIdx.x;
    int idx = block_i * blockDim.x + thread_i;
}

void cuda_ntt(LL *a,int n,LL MOD,bool invert) {
    bit_reverse(a, n);
    LL *a_d;
    cudaMalloc((void**)&a_d, sizeof(LL) * n);
    cudaMemcpy(a_d, a, sizeof(LL) * n, cudaMemcpyHostToDevice);

    for (int len = 2; len <= n; len <<= 1) {
        LL wn = qpow(ROOT, (MOD - 1) / len, MOD);
        if (invert) wn = qpow(wn, MOD - 2, MOD);
        LL* w = new LL[len / 2];
        w[0] = 1;
        for (int i = 1; i < len / 2; i++) w[i] = 1LL * wn * w[i - 1] % MOD;

        LL* w_d;
        cudaMalloc((void**)&w_d, sizeof(LL) * (len / 2));
        cudaMemcpy(w_d, w, sizeof(LL) * (len / 2), cudaMemcpyHostToDevice);
        int num_blocks = n/1024+1;
        int threads_per_block = 1024;
        partial_ntt<<<num_blocks, threads_per_block>>>(a_d, w_d, n,len, MOD);
        cudaDeviceSynchronize();
        // std::cout<<"len = "<<len<<std::endl;
        delete[] w;
        cudaFree(w_d);
        // std::cout<<"结束一次"<<std::endl;
    }
    // std::cout<<"开始完全"<<std::endl;
    cudaMemcpy(a, a_d, sizeof(LL) * n, cudaMemcpyDeviceToHost);
    cudaFree(a_d);
    // std::cout<<"完全结束"<<std::endl;

    if (invert) {
        LL inv_n = qpow(n, MOD - 2, MOD);
        for (int i = 0; i < n; i++) a[i] = 1LL * a[i] * inv_n % MOD;
    }
}

// #include "mul_cuda.h"
// #include "utils.h"

// // __global__ void partial_ntt(...)  -> 我们将修改这个kernel
// __global__ void partial_ntt(LL *a, LL*w, int n, int len, LL MOD){
//     int block_i = blockIdx.x;
//     int thread_j_start = threadIdx.x; // 每个线程的起始局部索引

//     // 计算这个block负责处理的数据段的基地址
//     int group_base_idx = block_i * len;

//     // 使用固定的线程数，通过循环处理一个group内的所有蝶形运算
//     // 每个线程的步长是 blockDim.x (block内的线程总数)
//     for (int j = thread_j_start; j < len / 2; j += blockDim.x) {
        
//         int current_idx = group_base_idx + j;
        
//         // 读取操作数
//         LL u = a[current_idx];
//         // 从w数组中读取对应的旋转因子，并计算v
//         LL v = (1LL * w[j] * a[current_idx + len/2]) % MOD;

//         // 执行蝶形运算并写回
//         a[current_idx] = (u + v) % MOD;
//         a[current_idx + len/2] = (u - v + MOD) % MOD;
//     }
// }


// void cuda_ntt(LL *a,int n,LL MOD,bool invert) {
//     bit_reverse(a, n);
//     LL *a_d;
//     cudaMalloc((void**)&a_d, sizeof(LL) * n);
//     cudaMemcpy(a_d, a, sizeof(LL) * n, cudaMemcpyHostToDevice);

//     for (int len = 2; len <= n; len <<= 1) {
//         LL wn = qpow(ROOT, (MOD - 1) / len, MOD);
//         if (invert) wn = qpow(wn, MOD - 2, MOD);
        
//         // 注意，这里的w数组大小是 len/2
//         LL* w = new LL[len / 2];
//         w[0] = 1;
//         for (int i = 1; i < len / 2; i++) w[i] = 1LL * wn * w[i - 1] % MOD;

//         LL* w_d;
//         cudaMalloc((void**)&w_d, sizeof(LL) * (len / 2));
//         cudaMemcpy(w_d, w, sizeof(LL) * (len / 2), cudaMemcpyHostToDevice);
        
//         // --- 这是修改的关键部分 ---
//         int num_blocks = n / len;
//         // 使用一个固定的、小于1024的线程数
//         int threads_per_block = 512; 
        
//         // 当需要处理的任务数(len/2)小于我们设定的线程数时，
//         // 没必要启动那么多线程，动态调整以节省资源
//         if (len / 2 < threads_per_block) {
//             threads_per_block = len / 2;
//         }

//         // 确保线程数至少为1 (在 len=2 的情况下, len/2=1)
//         if (threads_per_block == 0) threads_per_block = 1;

//         partial_ntt<<<num_blocks, threads_per_block>>>(a_d, w_d, n, len, MOD);
//         // --- 修改结束 ---

//         cudaDeviceSynchronize(); // 等待kernel执行完成
//         delete[] w;
//         cudaFree(w_d);
//     }
//     cudaMemcpy(a, a_d, sizeof(LL) * n, cudaMemcpyDeviceToHost);
//     cudaFree(a_d);

//     if (invert) {
//         LL inv_n = qpow(n, MOD - 2, MOD);
//         for (int i = 0; i < n; i++) a[i] = 1LL * a[i] * inv_n % MOD;
//     }
// }
void cuda_ntt_multiply(LL *a, LL *b, LL *ab, int n,LL MOD) {
    int size=1;
    while(size<2*n) size<<=1;
    for(int i=n;i<size;i++) a[i]=b[i]=0;
    cuda_ntt(a, size,MOD,false);
    cuda_ntt(b,size,MOD, false);
    for (int i = 0; i < size; i++) ab[i] = 1LL * a[i] * b[i] % MOD;
    cuda_ntt(ab,size,MOD, true);
}
