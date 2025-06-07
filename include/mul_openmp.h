#pragma once
typedef long long LL;

void openmp_ntt(LL *a,int n,LL MOD,bool invert=false);
void openmp_ntt_multiply(LL *a, LL *b, LL *ab, int n,LL MOD);
void openmp_crt_ntt_multiply(LL *a, LL *b, LL *ab, int n, LL p);

