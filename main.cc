#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sys/time.h>
#include <omp.h>
#include <arm_neon.h>
#include <cstdlib>  
#include <ctime> 
// 可以自行添加需要的头文件


const int ROOT = 3;
const int BITE = 63;
// const int MOD = 998244353;


void fRead(int *a, int *b, int *n, int *p, int input_id){
    // 数据输入函数
    std::string str1 = "/nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strin = str1 + str2 + ".in";
    char data_path[strin.size() + 1];
    std::copy(strin.begin(), strin.end(), data_path);
    data_path[strin.size()] = '\0';
    std::ifstream fin;
    fin.open(data_path, std::ios::in);
    fin>>*n>>*p;
    for (int i = 0; i < *n; i++){
        fin>>a[i];
    }
    for (int i = 0; i < *n; i++){   
        fin>>b[i];
    }
}

void fCheck(int *ab, int n, int input_id){
    // 判断多项式乘法结果是否正确
    std::string str1 = "/nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    char data_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), data_path);
    data_path[strout.size()] = '\0';
    std::ifstream fin;
    fin.open(data_path, std::ios::in);
    for (int i = 0; i < n * 2 - 1; i++){
        int x;
        fin>>x;
        if(x != ab[i]){
            std::cout<<"多项式乘法结果错误"<<std::endl;
            return;
        }
    }
    std::cout<<"多项式乘法结果正确"<<std::endl;
    return;
}

void fWrite(int *ab, int n, int input_id){
    // 数据输出函数, 可以用来输出最终结果, 也可用于调试时输出中间数组
    std::string str1 = "files/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    char output_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), output_path);
    output_path[strout.size()] = '\0';
    std::ofstream fout;
    fout.open(output_path, std::ios::out);
    for (int i = 0; i < n * 2 - 1; i++){
        fout<<ab[i]<<'\n';
    }
}

void poly_multiply(int *a, int *b, int *ab, int n, int p){
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            ab[i+j]=(1LL * a[i] * b[j] % p + ab[i+j]) % p;
        }
    }
}

int qpow(int a, int b, int p) {
    int ans = 1;
    while (b) {
        if (b & 1) ans = (1LL*ans * a) % p;
        a = (1LL*a * a) % p;
        b >>= 1;
    }
    return ans;
}

void bit_reverse(int *a, int n) {
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

void ntt(int *a,int n,int MOD,bool invert=false) {
    bit_reverse(a, n);

    for (int len = 2; len <= n; len <<= 1) {
        int wn = qpow(ROOT, (MOD - 1) / len, MOD);
        if (invert) wn = qpow(wn, MOD - 2, MOD);

        for (int i = 0; i < n; i += len) {
            int w = 1;
            for (int j = 0; j < len / 2; j++) {
                int u = a[i + j], v = 1LL * w * a[i + j + len / 2] % MOD;
                a[i + j] = (u + v) % MOD;
                a[i + j + len / 2] = (u - v+MOD) % MOD;
                w = 1LL * w * wn % MOD;
            }
        }
    }

    if (invert) {
        int inv_n = qpow(n, MOD - 2, MOD);
        for (int i = 0; i < n; i++) a[i] = 1LL * a[i] * inv_n % MOD;
    }
}

void ntt_multiply(int *a, int *b, int *ab, int n,int MOD) {
    int size=1;
    while(size<2*n) size<<=1;
    for(int i=n;i<size;i++) a[i]=b[i]=0;
    ntt(a, size,MOD,false);
    ntt(b,size,MOD, false);
    for (int i = 0; i < size; i++) ab[i] = 1LL * a[i] * b[i] % MOD;
    ntt(ab,size,MOD, true);
}

inline void neon_mod(int32x4_t *a, int64_t mu,int MOD){
    // int32x4_t mod = vdupq_n_s32(MOD);
    // while(true){
    //    uint32x4_t mask = vcgeq_s32(*a, mod);
    //    uint32_t flag = vmaxvq_u32(mask); 
    //    if(flag == 0) break;
    //    int32x4_t subtracted = vsubq_s32(*a, mod); 
    //    *a = vbslq_s32(mask, subtracted, *a); 
    // }
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


inline uint64_t vmaxvq_u64(uint64x2_t v) {
    uint64_t a = vgetq_lane_u64(v, 0);  
    uint64_t b = vgetq_lane_u64(v, 1);  
    return a > b ? a : b;               
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
    // a= uint64_t(a);
    // b= uint64_t(b);

    // uint32_t a_lo = (uint32_t)a;        // a的低32位
    // uint32_t a_hi = (uint32_t)(a >> 32); // a的高32位
    // uint32_t b_lo = (uint32_t)b;        // b的低32位
    // uint32_t b_hi = (uint32_t)(b >> 32); // b的高32位

    // uint32_t temp_a[2] = {a_lo, a_hi};
    // uint32_t temp_b[2] = {b_lo, b_hi};

    // uint32x2_t ua = vld1_u32(temp_a);
    // uint32x2_t ub = vld1_u32(temp_b);
    

    // // 使用NEON进行交叉计算
    // uint64x2_t result = vmull_u32(ua, ub);  // 初始化结果为0

    // // 低位：a_lo * b_lo
    // uint64_t low = (uint64_t)a_lo * b_lo;

    // // 中位：a_lo * b_hi + a_hi * b_lo
    // uint64_t mid = (uint64_t)a_lo * b_hi + (uint64_t)a_hi * b_lo;
    // uint32_t mid_hi = (uint32_t)(mid >> 32);  // 提取中位的高32位
    // uint32_t mid_lo = (uint32_t)mid;             // 提取中位的低32位


    // // 高位：a_hi * b_hi

    // // 处理中位：将低位结果左移32位，加上中位低32位，处理溢出
    // uint64_t lo = (uint64_t)mid_lo << 32;

    // // 如果低位溢出，进位到高位
    // uint64_t carry = (0xFFFFFFFFFFFFFFFF - lo < low);  // 溢出标志

    // uint64_t hi = (uint64_t)mid_hi+carry;
    // uint64x2_t temp = vcombine_u64(vcreate_u64(lo), vcreate_u64(hi));
    // result = vaddq_u64(result, temp);
    // return result;
}

inline void neon_mod_for_mul(int64x2_t *a, int MOD){
    // int64x2_t mod = vdupq_n_s64(MOD);
    // while(true){
    //    uint64x2_t mask = vcgeq_s64(*a, mod);
    //    uint64_t flag = vmaxvq_u64(mask); 
    //    if(flag == 0) break;
    //    int64x2_t subtracted = vsubq_s64(*a, mod); 
    //    *a = vbslq_s64(mask, subtracted, *a); 
    // }
    // int64x2_t mod = vdupq_n_s64(MOD);
    //    uint64x2_t mask = vcgeq_s64(*a, mod);
    //    int64x2_t subtracted = vsubq_s64(*a, mod); 
    //    *a = vbslq_s64(mask, subtracted, *a);
    // int i = 0;
    // while(true){
    //     int64x2_t mod = vdupq_n_s64((1LL << i) * MOD);
    //     uint64x2_t mask = vcgeq_s64(*a, mod);
    //     uint64_t flag = vmaxvq_u64(mask); 
    //     if(flag == 0)
    //         break;
    //     ++i;
    // }
    // if (i > 0) i--;
    // while(true){
    //    int64x2_t mod = vdupq_n_s64((1LL << i) *MOD);
    //    uint64x2_t mask = vcgeq_s64(*a, mod);
    //    uint64_t flag = vmaxvq_u64(mask); 
    //    if(flag){
    //     int64x2_t subtracted = vsubq_s64(*a, mod); 
    //     *a = vbslq_s64(mask, subtracted, *a); 
    //    }
    //    if(i==0) break;
    //    --i;
    // }
    int64x2_t mod = vdupq_n_s64(MOD);
    uint64x2_t mask = vcgeq_s64(*a, mod);
    int64x2_t subtracted = vsubq_s64(*a, mod); 
    *a = vbslq_s64(mask, subtracted, *a); 
    
}

inline void neon_barrett_reduction(int64x2_t *a, int64_t mu, int MOD){
    int64_t vals[2];
    int64_t subs[2];
    vst1q_s64(vals, *a);
    for(int i = 0; i < 2; i++){
        // int64_t q = (vals[i] * mu)>> BITE;
        // int64_t q = vals[i]/ MOD;
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
    // neon_mod_for_mul(a, MOD);
    // float64x2_t vals = vcvtq_f64_s64(*a);
    // float64x2_t mod_f = vdupq_n_f64(MOD);
    // int64x2_t sub = vcvtq_s64_f64(vmulq_f64(vdivq_f64(vals, mod_f), mod_f));
    // int64x2_t subs = vcvtq_s64_f64(vdivq_f64(vals, mod_f));
    // int64x2_t sub = vcvtq_s64_f64(vmulq_f64(vdivq_f64(vals, mod_f), mod_f));

    // for(int i = 0; i < 2; i++){
        // int64_t q = (vals[i] * mu)>> BITE;
        // int64_t q = vals[i]/ MOD;
        // subs[i] = q * MOD;        
    // }
    // int64x2_t sub = vld1q_s64(subs);

}


 inline int32x4_t neon_add(int32x4_t *a, int32x4_t *b, int MOD, int64_t MU) {
    int32x4_t c = vaddq_s32(*a, *b);
    neon_mod(&c, MU,MOD);
    return c;
    // int64x2_t add_lo = vaddl_s32(vget_low_s32(*a), vget_low_s32(*b));
    // int64x2_t add_hi = vaddl_s32(vget_high_s32(*a), vget_high_s32(*b));
    // neon_barrett_reduction(&add_lo, MU, MOD);
    // neon_barrett_reduction(&add_hi, MU, MOD);
    // return vcombine_s32(vmovn_s64(add_lo), vmovn_s64(add_hi));
}

inline int32x4_t neon_sub(int32x4_t *a, int32x4_t *b, int MOD, int64_t MU) {
    int32x4_t c = vsubq_s32(*a, *b);
    c= vaddq_s32(c, vdupq_n_s32(MOD));
    neon_mod(&c,MU,MOD);
    return c;
    // int64x2_t sub_lo = vsubl_s32(vget_low_s32(*a), vget_low_s32(*b));
    // int64x2_t sub_hi = vsubl_s32(vget_high_s32(*a), vget_high_s32(*b));
    
    // int64x2_t mod = vdupq_n_s64(MOD);
    // sub_lo = vaddq_s64(sub_lo, mod);
    // sub_hi = vaddq_s64(sub_hi, mod);

    // neon_barrett_reduction(&sub_lo, MU, MOD);
    // neon_barrett_reduction(&sub_hi, MU, MOD);
    
    // return vcombine_s32(vmovn_s64(sub_lo), vmovn_s64(sub_hi));
}

inline int32x4_t neon_mul(int32x4_t *a, int32x4_t *b, int MOD, int64_t MU) {
    // int32x4_t c = vmulq_s32(*a, *b);
    // neon_mod(&c, MOD);
    // return c;
    int64x2_t c_lo = vmull_s32(vget_low_s32(*a), vget_low_s32(*b));
    int64x2_t c_hi = vmull_s32(vget_high_s32(*a), vget_high_s32(*b));
    neon_barrett_reduction(&c_lo, MU, MOD);
    neon_barrett_reduction(&c_hi, MU, MOD);
    return vcombine_s32(vmovn_s64(c_lo), vmovn_s64(c_hi));
}

void neon_ntt(int *a,int n,int MOD,int64_t MU,bool invert=false) {
    bit_reverse(a, n);

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




// inline int mod_add(int a, int b, int MOD) {
//     int res = a + b;
//     if (res >= MOD) res -= MOD;
//     return res;
// }

// // 标准版模减
// inline int mod_sub(int a, int b, int MOD) {
//     int res = a - b;
//     if (res < 0) res += MOD;
//     return res;
// }

// // 标准版模乘
// inline int mod_mul(int a, int b, int MOD) {
//     return int((1LL * a * b) % MOD);
// }

// void test_neon_operations(int *save_correct_add, int *save_test_add,
//                            int *save_correct_sub, int *save_test_sub,
//                            int *save_correct_mul, int *save_test_mul,
//                            int N, int MOD) 
// {
//     srand(time(0));

//     for (int i = 0; i < N; i += 4) {
//         // 随机生成数据
//         int32_t a_scalar[4], b_scalar[4];
//         for (int j = 0; j < 4; ++j) {
//             a_scalar[j] = rand() % (2 * MOD);  // 范围可以大一点，测试循环减法
//             b_scalar[j] = rand() % (2 * MOD);
//         }

//         // 装载到向量
//         int32x4_t a_vec = vld1q_s32(a_scalar);
//         int32x4_t b_vec = vld1q_s32(b_scalar);

//         // ===== 1. 测试加法 =====
//         int32x4_t add_vec = neon_add(&a_vec, &b_vec, MOD);
//         vst1q_s32(save_test_add + i, add_vec); // 保存测试结果

//         for (int j = 0; j < 4; ++j) {
//             save_correct_add[i + j] = mod_add(a_scalar[j], b_scalar[j], MOD);
//         }

//         // ===== 2. 测试减法 =====
//         int32x4_t sub_vec = neon_sub(&a_vec, &b_vec, MOD);
//         vst1q_s32(save_test_sub + i, sub_vec); // 保存测试结果

//         for (int j = 0; j < 4; ++j) {
//             save_correct_sub[i + j] = mod_sub(a_scalar[j], b_scalar[j], MOD);
//         }

//         // ===== 3. 测试乘法 =====
//         int32x4_t mul_vec = neon_mul(&a_vec, &b_vec, MOD);
//         vst1q_s32(save_test_mul + i, mul_vec); // 保存测试结果

//         for (int j = 0; j < 4; ++j) {
//             save_correct_mul[i + j] = mod_mul(a_scalar[j], b_scalar[j], MOD);
//         }
//     }
// }



int a[300000], b[300000], ab[300000];
int main(int argc, char *argv[])
{
    
    // 保证输入的所有模数的原根均为 3, 且模数都能表示为 a \times 4 ^ k + 1 的形式
    // 输入模数分别为 7340033 104857601 469762049 263882790666241
    // 第四个模数超过了整型表示范围, 如果实现此模数意义下的多项式乘法需要修改框架
    // 对第四个模数的输入数据不做必要要求, 如果要自行探索大模数 NTT, 请在完成前三个模数的基础代码及优化后实现大模数 NTT
    // 输入文件共五个, 第一个输入文件 n = 4, 其余四个文件分别对应四个模数, n = 131072
    // 在实现快速数论变化前, 后四个测试样例运行时间较久, 推荐调试正确性时只使用输入文件 1
    int test_begin = 0;
    int test_end = 0;
    for(int i = test_begin; i <= test_end; ++i){
        long double ans = 0;
        int n_, p_;
        fRead(a, b, &n_, &p_, i);
        memset(ab, 0, sizeof(ab));
        auto Start = std::chrono::high_resolution_clock::now();
        // TODO : 将 poly_multiply 函数替换成你写的 ntt
        // poly_multiply(a, b, ab, n_, p_);
        // ntt_multiply(a, b, ab, n_, p_);
        neon_ntt_multiply(a, b, ab, n_, p_);
        auto End = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double,std::ratio<1,1000>>elapsed = End - Start;
        ans += elapsed.count();
        fCheck(ab, n_, i);
        std::cout<<"average latency for n = "<<n_<<" p = "<<p_<<" : "<<ans<<" (us) "<<std::endl;
        // 可以使用 fWrite 函数将 ab 的输出结果打印到 files 文件夹下
        // 禁止使用 cout 一次性输出大量文件内容
        fWrite(ab, n_, i);
    }

    // const int N = 12;   // 测试样本数量（一定是4的倍数）
    // const int MOD = 998244353; // 你自己设定

    // int save_correct_add[N], save_test_add[N];
    // int save_correct_sub[N], save_test_sub[N];
    // int save_correct_mul[N], save_test_mul[N];

    // test_neon_operations(save_correct_add, save_test_add,
    //                      save_correct_sub, save_test_sub,
    //                      save_correct_mul, save_test_mul,
    //                      N, MOD);
    
    // fWrite(save_correct_add,N/2,111);
    // fWrite(save_test_add,N/2,110);
    // fWrite(save_correct_sub,N/2,222);
    // fWrite(save_test_sub,N/2,220);
    // fWrite(save_correct_mul,N/2,333);
    // fWrite(save_test_mul,N/2,330);
    
    // return 0;
}


