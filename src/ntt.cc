#include "ntt.h"

#include "utils.h"


void poly_multiply(int *a, int *b, int *ab, int n, int p){
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            ab[i+j]=(1LL * a[i] * b[j] % p + ab[i+j]) % p;
        }
    }
}



void ntt(LL *a,int n,LL MOD,bool invert) {
    bit_reverse(a, n);

    for (int len = 2; len <= n; len <<= 1) {
        LL wn = qpow(ROOT, (MOD - 1) / len, MOD);
        if (invert) wn = qpow(wn, MOD - 2, MOD);

        for (int i = 0; i < n; i += len) {
            LL w = 1;
            for (int j = 0; j < len / 2; j++) {
                LL u = a[i + j], v = 1LL * w * a[i + j + len / 2] % MOD;
                a[i + j] = (u + v) % MOD;
                a[i + j + len / 2] = (u - v+MOD) % MOD;
                w = 1LL * w * wn % MOD;
            }
        }
    }

    if (invert) {
        LL inv_n = qpow(n, MOD - 2, MOD);
        for (int i = 0; i < n; i++) a[i] = 1LL * a[i] * inv_n % MOD;
    }
}

void ntt_multiply(LL *a, LL *b, LL *ab, int n,LL MOD) {
    int size=1;
    while(size<2*n) size<<=1;
    for(int i=n;i<size;i++) a[i]=b[i]=0;
    ntt(a, size,MOD,false);
    ntt(b,size,MOD, false);
    for (int i = 0; i < size; i++) ab[i] = 1LL * a[i] * b[i] % MOD;
    ntt(ab,size,MOD, true);
}
