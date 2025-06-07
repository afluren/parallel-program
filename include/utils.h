#pragma once
typedef long long LL;
constexpr int BITE = 63;
constexpr int ROOT = 3;
const int NUM_THREADS = 4;

LL qpow(LL a, LL b, LL p);

void bit_reverse(LL *a, int n);

LL extend_gcd(LL a, LL b, LL &x, LL &y);

LL inv(LL a, LL p);

inline LL mulmod ( LL a, LL b, LL p )  {
    unsigned long long x = 1ULL * a%p, y = 1ULL * b%p;
    LL result = x * y - (LL)((long double)x / p * y + 0.5) * p;
    return result < 0 ? result + p : result;
}

void CRT(LL *ab1, LL *ab2, int n, int p1, int p2);