#pragma once
typedef long long LL;

void poly_multiply(int *a, int *b, int *ab, int n, int p);
void ntt(LL *a,int n,LL MOD,bool invert=false);
void ntt_multiply(LL *a, LL *b, LL *ab, int n,LL MOD);
