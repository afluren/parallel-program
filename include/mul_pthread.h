#pragma once
typedef long long LL;

void *partial_ntt(void *arg);

void pthread_ntt(LL *a, int n, LL MOD, bool invert = false);

void pthread_ntt_multiply(LL *a, LL *b, LL *ab, int n, LL MOD);

void* pthread_ntt_multiply_thread(void *arg);

void pthread_crt_ntt_multiply(LL *a, LL *b, LL *ab, int n, LL p);