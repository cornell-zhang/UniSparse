#include <random>
#include <chrono>
#include "taco.h"
using namespace taco;
int main(int argc, char* argv[]) {
  std::default_random_engine gen(0);
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  Format dcsr({Sparse, Sparse});
  Format  dv({Dense});

  char *file_name = argv[1];
  auto t1 = std::chrono::high_resolution_clock::now();
  Tensor<double> A = read(file_name, dcsr);
  auto t2 = std::chrono::high_resolution_clock::now();
  double convert_time = double(std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()) / 1000000;
  std::cout << "Convert to DCSR time : " << convert_time << " seconds" << std::endl;

  Tensor<double> x({A.getDimension(1)}, dv);
  for (int i = 0; i < x.getDimension(0); ++i) {
    x.insert({i}, unif(gen));
  }
  x.pack();

  Tensor<double> y({A.getDimension(0)}, dv);

  IndexVar i, j;
  y(i) = A(i,j) * x(j);

  y.compile();
  y.assemble();

  auto t3 = std::chrono::high_resolution_clock::now();
  y.compute();
  auto t4 = std::chrono::high_resolution_clock::now();
  double compute_time = double(std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count()) / 1000000;
  std::cout << "Compute DCSR SpMV time : " << compute_time << " seconds" << std::endl;

  write("y3.tns", y);
}
