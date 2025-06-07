#include "mul_pthread.h"

#include <algorithm>

#include <pthread.h>

#include "utils.h"
#include "ntt.h"


// 同步原语
pthread_mutex_t mutex;
pthread_barrier_t barrier;

// 全局共享数据
LL *global_w;
LL global_MOD;
int global_len;
bool should_exit = false;

struct partial_ntt_arg {
  LL *a;
  int start;
  int end;
  int thread_id;
};

void *partial_ntt(void *arg) {
  partial_ntt_arg *narg = (partial_ntt_arg *)arg;
  while (true) {
    pthread_barrier_wait(&barrier); // 等待所有线程都到达此处
    pthread_mutex_lock(&mutex);
    if (should_exit) {
      pthread_mutex_unlock(&mutex);
      break;
    }
    // 复制全局数据到本地
    LL *w = global_w;
    int len = global_len;
    int start = narg->start;
    int end = narg->end;
    LL MOD = global_MOD;

    pthread_mutex_unlock(&mutex);
    // 执行NTT计算
    for (int i = start; i < end; i += len) {
      for (int j = 0; j < len / 2; j++) {
        LL u = narg->a[i + j];
        LL v = (1LL * w[j] * narg->a[i + j + len / 2]) % MOD;
        narg->a[i + j] = (u + v) % MOD;
        narg->a[i + j + len / 2] = (u - v + MOD) % MOD;
      }
    }
    pthread_barrier_wait(&barrier);
  }
  return NULL;
}

void pthread_ntt(LL *a, int n, LL MOD, bool invert) {
  bit_reverse(a, n);

  pthread_t threads[NUM_THREADS];
  partial_ntt_arg args[NUM_THREADS];
  pthread_barrier_init(&barrier, NULL, NUM_THREADS + 1);
  pthread_mutex_init(&mutex, NULL);
  for (int i = 0; i < NUM_THREADS; i++) {
    args[i].a = a;
    args[i].thread_id = i;
    int iRet = pthread_create(&threads[i], NULL, partial_ntt, (void *)&args[i]);
    if (iRet != 0) {
      return;
    }
  }

  global_MOD = MOD;
  const LL ROOT = 3;

  for (int len = 2; len <= n; len <<= 1) {
    LL wn = qpow(ROOT, (MOD - 1) / len, MOD);
    if (invert)
      wn = qpow(wn, MOD - 2, MOD);

    LL *w = new LL[len / 2];
    w[0] = 1;
    for (int i = 1; i < len / 2; i++) {
      w[i] = (1LL * wn * w[i - 1]) % MOD;
    }

    global_w = w;
    global_len = len;

    // 计算每个线程的工作范围
    int total_blocks = n / len;
    int blocks_per_thread = total_blocks / NUM_THREADS;
    int extra_blocks = total_blocks % NUM_THREADS;

    int current_start = 0;
    for (int i = 0; i < NUM_THREADS; i++) {
      args[i].start = current_start;
      int blocks_for_this_thread =
          blocks_per_thread + (i < extra_blocks ? 1 : 0);
      args[i].end = current_start + blocks_for_this_thread * len;
      current_start = args[i].end;
    }
    pthread_barrier_wait(&barrier); // 开启所有线程

    pthread_barrier_wait(&barrier); // 等待所有线程都退出

    delete[] w;
  }

  pthread_mutex_lock(&mutex);
  should_exit = true;
  pthread_mutex_unlock(&mutex);
  pthread_barrier_wait(&barrier);
  // 等待所有线程结束
  for (int i = 0; i < NUM_THREADS; i++) {
    pthread_join(threads[i], NULL);
  }

  should_exit = false;

  pthread_mutex_destroy(&mutex);
  pthread_barrier_destroy(&barrier);

  // 如果是逆变换，需要除以n
  if (invert) {
    LL inv_n = qpow(n, MOD - 2, MOD);
    for (int i = 0; i < n; i++) {
      a[i] = (1LL * a[i] * inv_n) % MOD;
    }
  }
}

void pthread_ntt_multiply(LL *a, LL *b, LL *ab, int n, LL MOD) {
  int size = 1;
  while (size < 2 * n)
    size <<= 1;
  for (int i = n; i < size; i++)
    a[i] = b[i] = 0;
  pthread_ntt(a, size, MOD, false);
  pthread_ntt(b, size, MOD, false);
  for (int i = 0; i < size; i++)
    ab[i] = 1LL * a[i] * b[i] % MOD;
  pthread_ntt(ab, size, MOD, true);
}

struct ntt_arg{
    LL *a;
    LL *b;
    LL *ab;
    int n;
    LL p; // 和MOD一样
};

void* pthread_ntt_multiply_thread(void *arg) {
    ntt_arg *narg = (ntt_arg *)arg;
    LL *a = narg->a;
    LL *b = narg->b;
    LL *ab = narg->ab;
    int n = narg->n;
    LL MOD = narg->p;
    int size=1;
    while(size<2*n) size<<=1;
    for(int i=n;i<size;i++) a[i]=b[i]=0;
    ntt(a, size,MOD,false);
    ntt(b,size,MOD, false);
    for (int i = 0; i < size; i++) ab[i] = 1LL * a[i] * b[i] % MOD;
    ntt(ab,size,MOD, true);
    pthread_exit(NULL);
}


void pthread_crt_ntt_multiply(LL *a, LL *b, LL *ab, int n, LL p){
    // 计算长度，用于创建复制变量
    int size=1;
    while(size<2*n) size<<=1;
    LL *a_copy[4];
    LL *b_copy[4];
    LL *ab_copy[4];
    struct ntt_arg arg[4];
    //TODO：寻找合适的模数，能找到四个都是3的吗......还真能找到：
    // https://blog.miskcoo.com/2014/07/fft-prime-table 记录了常用的素数及其原根，致谢@miskcoo
    LL ntt_p[4] = {167772161,469762049,998244353,1004535809};
    a_copy[0] = a;
    b_copy[0] = b;
    ab_copy[0] = ab;
    for(int i = 1; i <= 3; i++){
        a_copy[i] = new LL[size];
        b_copy[i] = new LL[size];
        ab_copy[i] = new LL[size];
    }
    for(int i = 1; i<=3; i++){
        std::copy(a, a + size, a_copy[i]);
        std::copy(b, b + size, b_copy[i]);
        std::fill(ab_copy[i], ab_copy[i] + size, 0);
    }
    for(int i=0; i<=3; i++){
        arg[i].a = a_copy[i];
        arg[i].b = b_copy[i];
        arg[i].ab = ab_copy[i];
        arg[i].n = n;
        arg[i].p = ntt_p[i];
    }
    void *status;
    pthread_t threads[NUM_THREADS];
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    for(int i = 0; i < NUM_THREADS; i++){
        pthread_create(&threads[i], &attr, pthread_ntt_multiply_thread,(void *) &arg[i]);
    }
    pthread_attr_destroy(&attr);
    for(int i = 0; i < NUM_THREADS; i++){
        pthread_join(threads[i], &status);
    }
    for(int i=0;i<2;i++){
        CRT(ab_copy[2*i], ab_copy[2*i+1], size, ntt_p[2*i], ntt_p[2*i+1]);
    }

    LL p1 = 1LL * ntt_p[0] * ntt_p[1];
    LL p2 = 1LL * ntt_p[2] * ntt_p[3];
    LL inv_p2 = inv(p2, p1);
    for(int i=0;i<size;i++){
        LL k = mulmod(((ab_copy[0][i]-ab_copy[2][i])% p1 + p1)%p1 , (inv_p2 % p1) , p1);
        LL temp= (mulmod(k, p2, p) + ab_copy[2][i])%p;
        ab[i] = temp;
    }
    for(int i=1;i<=3;i++){
        delete[] a_copy[i];
        delete[] b_copy[i];
        delete[] ab_copy[i];
    }
}