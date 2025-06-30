#pragma once
typedef long long LL;

void cuda_ntt(LL *a, int n, LL MOD, bool invert = false);
void cuda_ntt_multiply(LL *a, LL *b, LL *ab, int n, LL MOD);
