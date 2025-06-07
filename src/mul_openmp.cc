#include "mul_openmp.h"

#include <algorithm>

#include <omp.h>

#include "utils.h"
#include "ntt.h"

void openmp_ntt(LL *a,int n,LL MOD,bool invert) {
    bit_reverse(a, n);

    for (int len = 2; len <= n; len <<= 1) {
        LL wn = qpow(ROOT, (MOD - 1) / len, MOD);
        if (invert) wn = qpow(wn, MOD - 2, MOD);
        LL* w = new LL[len / 2];
        w[0] = 1;
        for (int i = 1; i < len / 2; i++) w[i] = 1LL * wn * w[i - 1] % MOD;

        omp_set_num_threads(NUM_THREADS);
        #pragma omp parallel for 
        for (int i = 0; i < n; i += len) {
            
            for (int j = 0; j < len / 2; j++) {
                LL u = a[i + j], v = 1LL * w[j] * a[i + j + len / 2] % MOD;
                a[i + j] = (u + v) % MOD;
                a[i + j + len / 2] = (u - v+MOD) % MOD;
            }
        }
    }

    if (invert) {
        LL inv_n = qpow(n, MOD - 2, MOD);
        for (int i = 0; i < n; i++) a[i] = 1LL * a[i] * inv_n % MOD;
    }
}

void openmp_ntt_multiply(LL *a, LL *b, LL *ab, int n,LL MOD) {
    int size=1;
    while(size<2*n) size<<=1;
    for(int i=n;i<size;i++) a[i]=b[i]=0;
    openmp_ntt(a, size,MOD,false);
    openmp_ntt(b,size,MOD, false);
    for (int i = 0; i < size; i++) ab[i] = 1LL * a[i] * b[i] % MOD;
    openmp_ntt(ab,size,MOD, true);
}

void openmp_crt_ntt_multiply(LL *a, LL *b, LL *ab, int n, LL p){
    // 计算长度，用于创建复制变量
    int size=1;
    while(size<2*n) size<<=1;
    LL *a_copy[4];
    LL *b_copy[4];
    LL *ab_copy[4];
    //TODO：寻找合适的模数，能找到四个都是3的吗......还真能找到：
    // https://blog.miskcoo.com/2014/07/fft-prime-table 记录了常用的素数及其原根，致谢@miskcoo
    LL ntt_p[4] = {167772161,469762049,998244353,1004535809};
    a_copy[0] = a;
    b_copy[0] = b;
    ab_copy[0] = ab;
    for(int i = 1; i <= 3; i++){
        a_copy[i] = new LL[size];
        b_copy[i] = new LL[size];
        ab_copy[i] = new LL[size];
    }
    for(int i = 1; i<=3; i++){
        std::copy(a, a + size, a_copy[i]);
        std::copy(b, b + size, b_copy[i]);
        std::fill(ab_copy[i], ab_copy[i] + size, 0);
    }
    #pragma omp parallel for num_threads(NUM_THREADS)
    for(int i=0; i<=3; i++){
        ntt_multiply(a_copy[i], b_copy[i], ab_copy[i], n, ntt_p[i]);
    }
    for(int i=0;i<2;i++){
        CRT(ab_copy[2*i], ab_copy[2*i+1], size, ntt_p[2*i], ntt_p[2*i+1]);
    }

    LL p1 = 1LL * ntt_p[0] * ntt_p[1];
    LL p2 = 1LL * ntt_p[2] * ntt_p[3];
    LL inv_p2 = inv(p2, p1);
    #pragma omp parallel for num_threads(NUM_THREADS)
    for(int i=0;i<size;i++){
        LL k = mulmod(((ab_copy[0][i]-ab_copy[2][i])% p1 + p1)%p1 , (inv_p2 % p1) , p1);
        LL temp= (mulmod(k, p2, p) + ab_copy[2][i])%p;
        ab[i] = temp;
    }
    for(int i=1;i<=3;i++){
        delete[] a_copy[i];
        delete[] b_copy[i];
        delete[] ab_copy[i];
    }
}

