#include "utils.h"

#include <iostream>

LL qpow(LL a, LL b, LL p) {
    LL ans = 1;
    while (b) {
        if (b & 1) ans = (1LL*ans * a) % p;
        a = (1LL*a * a) % p;
        b >>= 1;
    }
    return ans;
}

void bit_reverse(LL *a, int n) {
    int j = 0;
    for (int i = 1; i < n; i++) {
        int bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
}

LL extend_gcd(LL a, LL b, LL &x, LL &y) {
    if (b == 0) {
        x = 1;
        y = 0;
        return a;
    }
    LL x1, y1;
    LL d = extend_gcd(b, a % b, x1, y1);
    x = y1;
    y = x1 - (a / b) * y1;
    return d;    
}

LL inv(LL a, LL p) {
    LL x, y;
    LL d = extend_gcd(a, p, x, y);
    if (d != 1) return -1;
    return (x % p + p) % p;
}

void CRT(LL *ab1, LL *ab2, int n, int p1, int p2){
    /*
    * 中国剩余定理合并，最终计算结果会保存在ab1里
    * @param n为多项式长度
    * @param p1,p2为两个多项式的模数
    * @param ab1,ab2为两个多项式的系数
    */
    LL P = 1LL * p1 * p2;
    LL P1 = P / p1;
    LL P2 = P / p2;  
    LL inv_P1 = inv(P1, (LL)(p1));
    LL inv_P2 = inv(P2, (LL)(p2));
    LL temp1 = inv_P1 % P2 * P1;
    LL temp2 = inv_P2 % P1 * P2;
    LL m1 = temp1 / P1;
    LL m2 = temp2 / P2;
    // # pragma omp parallel for num_threads(NUM_THREADS)
    for(int i = 0; i < n; i++){
        ab1[i] = (1LL * m1 * ab1[i] % P2 * P1 + (1LL * m2 * ab2[i] % P1 * P2)) % P;
    }
}