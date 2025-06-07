#pragma once

#include <arm_neon.h>

void neon_mod(int32x4_t *a, int64_t mu,int MOD);


inline uint64_t vmaxvq_u64(uint64x2_t v) {
    uint64_t a = vgetq_lane_u64(v, 0);  
    uint64_t b = vgetq_lane_u64(v, 1);  
    return a > b ? a : b;               
}

uint64x2_t neon_mul_int64(int64_t a, int64_t b);

inline void neon_mod_for_mul(int64x2_t *a, int MOD){
    int64x2_t mod = vdupq_n_s64(MOD);
    uint64x2_t mask = vcgeq_s64(*a, mod);
    int64x2_t subtracted = vsubq_s64(*a, mod); 
    *a = vbslq_s64(mask, subtracted, *a); 
    
}

void neon_barrett_reduction(int64x2_t *a, int64_t mu, int MOD);


inline int32x4_t neon_add(int32x4_t *a, int32x4_t *b, int MOD, int64_t MU) {
    int32x4_t c = vaddq_s32(*a, *b);
    neon_mod(&c, MU,MOD);
    return c;
}

inline int32x4_t neon_sub(int32x4_t *a, int32x4_t *b, int MOD, int64_t MU) {
    int32x4_t c = vsubq_s32(*a, *b);
    c= vaddq_s32(c, vdupq_n_s32(MOD));
    neon_mod(&c,MU,MOD);
    return c;
}

inline int32x4_t neon_mul(int32x4_t *a, int32x4_t *b, int MOD, int64_t MU) {
    int64x2_t c_lo = vmull_s32(vget_low_s32(*a), vget_low_s32(*b));
    int64x2_t c_hi = vmull_s32(vget_high_s32(*a), vget_high_s32(*b));
    neon_barrett_reduction(&c_lo, MU, MOD);
    neon_barrett_reduction(&c_hi, MU, MOD);
    return vcombine_s32(vmovn_s64(c_lo), vmovn_s64(c_hi));
}

void neon_ntt(int *a,int n,int MOD,int64_t MU,bool invert=false);

void neon_ntt_multiply(int *a, int *b, int *ab, int n,int MOD);
