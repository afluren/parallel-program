#ifndef UTILS_H
#define UTILS_H

typedef long long LL;

LL qpow(LL a, LL b, LL p);

void bit_reverse(LL *a, int n);

LL extend_gcd(LL a, LL b, LL &x, LL &y);

LL inv(LL a, LL p);

inline LL mulmod ( LL a, LL b, LL p )  {
    unsigned long long x = 1ULL * a%p, y = 1ULL * b%p;
    LL result = x * y - (LL)((long double)x / p * y + 0.5) * p;
    return result < 0 ? result + p : result;
}

#endif