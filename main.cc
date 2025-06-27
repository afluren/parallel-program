#include <limits.h>
#include <mpi.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "mul_openmp.h"
#include "mul_pthread.h"
#include "ntt.h"
#include "simd.h"
#include "utils.h"

// 可以自行添加需要的头文件
typedef long long LL;

// const int MOD = 998244353;

void fRead(LL *a, LL *b, int *n, LL *p, int input_id) {
  // 数据输入函数
  std::string str1 = "./nttdata/";
  std::string str2 = std::to_string(input_id);
  std::string strin = str1 + str2 + ".in";
  char data_path[strin.size() + 1];
  std::copy(strin.begin(), strin.end(), data_path);
  data_path[strin.size()] = '\0';
  std::ifstream fin;
  fin.open(data_path, std::ios::in);
  fin >> *n >> *p;
  for (int i = 0; i < *n; i++) {
    fin >> a[i];
  }
  for (int i = 0; i < *n; i++) {
    fin >> b[i];
  }
}

void fCheck(LL *ab, int n, int input_id) {
  // 判断多项式乘法结果是否正确
  std::string str1 = "./nttdata/";
  std::string str2 = std::to_string(input_id);
  std::string strout = str1 + str2 + ".out";
  char data_path[strout.size() + 1];
  std::copy(strout.begin(), strout.end(), data_path);
  data_path[strout.size()] = '\0';
  std::ifstream fin;
  fin.open(data_path, std::ios::in);
  for (int i = 0; i < n * 2 - 1; i++) {
    long long x;
    fin >> x;
    if (x != ab[i]) {
      std::cout << "多项式乘法结果错误" << std::endl;
      return;
    }
  }
  std::cout << "多项式乘法结果正确" << std::endl;
  return;
}

void fWrite(LL *ab, int n, int input_id) {
  // 数据输出函数, 可以用来输出最终结果, 也可用于调试时输出中间数组
  std::string str1 = "files/";
  std::string str2 = std::to_string(input_id);
  std::string strout = str1 + str2 + ".out";
  char output_path[strout.size() + 1];
  std::copy(strout.begin(), strout.end(), output_path);
  output_path[strout.size()] = '\0';
  std::ofstream fout;
  fout.open(output_path, std::ios::out);
  for (int i = 0; i < n * 2 - 1; i++) {
    fout << ab[i] << '\n';
  }
}

LL a[300000], b[300000], ab[300000];
static LL ab1[300000], ab2[300000], ab3[300000];
int main(int argc, char *argv[]) {

  // 保证输入的所有模数的原根均为 3, 且模数都能表示为 a \times 4 ^ k + 1 的形式
  // 输入模数分别为 7340033 104857601 469762049 1337006139375617
  // 第四个模数超过了整型表示范围, 如果实现此模数意义下的多项式乘法需要修改框架
  // 对第四个模数的输入数据不做必要要求, 如果要自行探索大模数 NTT,
  // 请在完成前三个模数的基础代码及优化后实现大模数 NTT 输入文件共五个,
  // 第一个输入文件 n = 4, 其余四个文件分别对应四个模数, n = 131072
  // 在实现快速数论变化前, 后四个测试样例运行时间较久,
  // 推荐调试正确性时只使用输入文件 1
  int provide;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provide);
  int rank, world_sz;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_sz);
  if (world_sz != 4) {
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  int test_begin = 0;
  int test_end = 4;
  LL ntt_p[4] = {167772161, 469762049, 998244353, 1004535809};
  for (int i = test_begin; i <= test_end; ++i) {
    MPI_Barrier(MPI_COMM_WORLD);
    double ans = MPI_Wtime();
    int n_;
    LL p_;
    memset(ab, 0, sizeof(ab));
    if (rank == 0) {
      fRead(a, b, &n_, &p_, i);
    }
    MPI_Bcast(&n_, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&p_, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(a, 300000, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, 300000, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    ntt_multiply(a, b, ab, n_, ntt_p[rank]);
    if (rank != 0) {
      MPI_Send(ab, 300000, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    }
    if (rank == 0) {
      MPI_Recv(ab1, 300000, MPI_LONG_LONG, 1, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Recv(ab2, 300000, MPI_LONG_LONG, 2, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      MPI_Recv(ab3, 300000, MPI_LONG_LONG, 3, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      int size = 1;
      while (size < 2 * n_)
        size <<= 1;
      openmp_CRT(ab, ab1, size, ntt_p[0], ntt_p[1]);
      openmp_CRT(ab2, ab3, size, ntt_p[2], ntt_p[3]);
      LL p1 = 1LL * ntt_p[0] * ntt_p[1];
      LL p2 = 1LL * ntt_p[2] * ntt_p[3];
      LL inv_p2 = inv(p2, p1);
#pragma omp parallel for num_threads(2)
      for (int i = 0; i < size; i++) {
        LL k = mulmod(((ab[i] - ab2[i]) % p1 + p1) % p1, (inv_p2 % p1), p1);
        LL temp = (mulmod(k, p2, p_) + ab2[i]) % p_;
        ab[i] = temp;
      }
      fCheck(ab, n_, i);
      ans = MPI_Wtime() - ans;
      std::cout << "average latency for n = " << n_ << " p = " << p_ << " : "
                << ans << " (s) " << std::endl;
    }
  }
  MPI_Finalize();
  return 0;
}
