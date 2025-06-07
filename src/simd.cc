#include "simd.h"

#include <arm_neon.h>

#include "utils.h"



void neon_mod(int32x4_t *a, int64_t mu,int MOD){
    int32_t vals[4];
    int32_t subs[4];
    vst1q_s32(vals, *a);
    for(int i = 0; i < 4; i++){
        int64_t q = (vals[i] * mu)>> BITE;
        // int64_t q = vals[i]/ MOD;
        subs[i] = static_cast<int32_t>(q * MOD);        
    }
    int32x4_t sub = vld1q_s32(subs);
    *a = vsubq_s32(*a, sub);

    int i = 0;
    while(true){
        int32x4_t mod = vdupq_n_s32((1 << i) * MOD);
        uint32x4_t mask = vcgeq_s32(*a, mod);
        uint32_t flag = vmaxvq_u32(mask); 
        if(flag == 0)
            break;
        ++i;
    }
    if (i > 0) i--;
    while(true){
       int32x4_t mod = vdupq_n_s32((1 << i) *MOD);
       uint32x4_t mask = vcgeq_s32(*a, mod);
       uint32_t flag = vmaxvq_u32(mask); 
       if(flag){
        int32x4_t subtracted = vsubq_s32(*a, mod); 
        *a = vbslq_s32(mask, subtracted, *a); 
       }
       if(i==0) break;
       --i;
    }
}

uint64x2_t neon_mul_int64(int64_t a, int64_t b) {
    a= uint64_t(a);
    b= uint64_t(b);

    uint32_t a_lo = (uint32_t)a;   
    uint32_t a_hi = (uint32_t)(a >> 32); 
    uint32_t b_lo = (uint32_t)b;     
    uint32_t b_hi = (uint32_t)(b >> 32); 

    uint64x2_t result = vdupq_n_u64(0); 

    uint64_t low = (uint64_t)a_lo * b_lo;


    uint64_t mid = (uint64_t)a_lo * b_hi + (uint64_t)a_hi * b_lo;
    uint32_t mid_hi = (uint32_t)(mid >> 32);  
    uint32_t mid_lo = (uint32_t)mid;           

    uint64_t high = (uint64_t)a_hi * b_hi;


    uint64_t lo = (uint64_t)mid_lo << 32;
    uint64_t final_low = low + lo;

    uint64_t carry = (0xFFFFFFFFFFFFFFFF - lo < low);  

    uint64_t final_high = high + (mid_hi) + carry;

    result = vsetq_lane_u64(final_low, result, 0); 
    result = vsetq_lane_u64(final_high, result, 1); 
    return result;
}



void neon_barrett_reduction(int64x2_t *a, int64_t mu, int MOD){
    int64_t vals[2];
    int64_t subs[2];
    vst1q_s64(vals, *a);
    for(int i = 0; i < 2; i++){
        uint64x2_t q_temp = neon_mul_int64(vals[i], mu);
        uint64_t q_lo = vgetq_lane_u64(q_temp, 0);
        uint64_t q_hi = vgetq_lane_u64(q_temp, 1);
        int64_t q = (q_lo >> BITE) + (q_hi << (64 - BITE));
        subs[i] = q * MOD;        
    }
    int64x2_t sub = vld1q_s64(subs);
    *a = vsubq_s64(*a, sub);
    int64x2_t mod = vdupq_n_s64(MOD);
    uint64x2_t mask = vcgeq_s64(*a, mod);
    int64x2_t subtracted = vsubq_s64(*a, mod); 
    *a = vbslq_s64(mask, subtracted, *a); 

}


void neon_ntt(int *a,int n,int MOD,int64_t MU,bool invert) {
    // bit_reverse(a, n);

    for (int len = 2; len <= n; len <<= 1) {
        int wn = qpow(ROOT, (MOD - 1) / len, MOD);
        if (invert) wn = qpow(wn, MOD - 2, MOD);

        for (int i = 0; i < n; i += len) {
            int w = 1;
            for (int j = 0; j < len / 2; j+=4) {
                if(j+3<len/2){
                    int32_t temp[4];
                    for (int k = 0; k < 4; k++) {
                        temp[k] = w;
                        w = 1LL * w * wn % MOD;
                    }
                    int32x4_t ww = vld1q_s32(temp);  
                    int32x4_t u = vld1q_s32(a + i + j);
                    int32x4_t v = vld1q_s32(a + i + j + len / 2);
                    int32x4_t wv = neon_mul(&ww, &v, MOD,MU);

                    int32x4_t u_add_v = neon_add(&u, &wv, MOD,MU);
                    int32x4_t u_sub_v = neon_sub(&u, &wv, MOD,MU);

                    vst1q_s32(a + i + j, u_add_v);
                    vst1q_s32(a + i + j + len / 2, u_sub_v);
                }
                else{
                    for(int k = j; k < len/2; k++){
                        int u = a[i + k], v = 1LL * w * a[i + k + len / 2] % MOD;
                        a[i + k] = (u + v) % MOD;
                        a[i + k + len / 2] = (u - v+MOD) % MOD;
                        w = 1LL * w * wn % MOD;
                    }
                }
            }
        }
    }

    if (invert) {
        int inv_n = qpow(n, MOD - 2, MOD);
        for (int i = 0; i < n; i += 4) {
          int32x4_t a4 = vld1q_s32(a + i);
          int32x4_t inv_n4 = vdupq_n_s32(inv_n);
          int32x4_t ab4 = neon_mul(&a4, &inv_n4, MOD,MU);
          vst1q_s32(a + i, ab4);
        //   a[i] = 1LL * a[i] * inv_n % MOD;
        }
    }
}

void neon_ntt_multiply(int *a, int *b, int *ab, int n,int MOD) {
    const int64_t MU = (1LL << BITE) / static_cast<int64_t>(MOD);
    int size=1;
    while(size<2*n) size<<=1;
    for(int i=n;i<size;i++) a[i]=b[i]=0;
    neon_ntt(a, size,MOD,MU,false);
    neon_ntt(b,size,MOD,MU,false);
    for (int i = 0; i < size; i+=4) {
        // ab[i] = 1LL * a[i] * b[i] % MOD;
      int32x4_t a4 = vld1q_s32(a + i);
      int32x4_t b4 = vld1q_s32(b + i);
      int32x4_t ab4 = neon_mul(&a4, &b4, MOD,MU);
      vst1q_s32(ab + i, ab4);
    }
    neon_ntt(ab,size,MOD,MU,true);
}