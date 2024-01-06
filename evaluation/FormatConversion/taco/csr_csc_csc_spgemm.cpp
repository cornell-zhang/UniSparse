#include <random>
#include <chrono>
#include "taco.h"
using namespace taco;
int main(int argc, char* argv[]) {

  Format csr({Dense, Sparse});
  Format csc({Dense, Sparse},{1, 0});

  char *file_name0 = argv[1];
  char *file_name1 = argv[2];
  Tensor<double> A = read(file_name0, csr);
  Tensor<double> B = read(file_name1, csc);

  Tensor<double> x({A.getDimension(0), B.getDimension(1)}, csc);

  x.pack();

  IndexVar i, j, k;
  x(i, j) = A(i, k) * B(k, j);

  x.compile();
  
  auto t3 = std::chrono::high_resolution_clock::now();
  x.assemble();
  x.compute();
  auto t4 = std::chrono::high_resolution_clock::now();
  double compute_time = double(std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count()) / 1000000;
  std::cout << "Compute CSR SpGEMM time : " << compute_time << " seconds" << std::endl;

//  write("y.tns", y);
}
