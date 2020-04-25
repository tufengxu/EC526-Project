#include "tsne.cpp"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

MatrixXd load_iris();
MatrixXd load_mnist();

const std::string result_path = "tsne_mnist.dat";

int main() {
    std::cout << "# Reading Data ..." << std::endl;
    auto data_mat = load_mnist();

    std::cout << "# tSNE Start ..." << std::endl;
    MatrixXd result = tSNE(data_mat, 2, 50, 20.0);
    
    std::ofstream data_store(result_path, std::ios::trunc);
    
    data_store << result;

    return 0;
}

MatrixXd load_iris() {
  std::ifstream iris_data("iris.data");
  MatrixXd iris_mat(150, 4);
  double temp = 0;
  std::string temp_str;
  std::vector<double> iris_vec;
  while (std::getline(iris_data, temp_str)) {
    std::stringstream temp_ss(temp_str);
    std::vector<std::string> tokens;
    while (temp_ss.good()) {
      std::string temp_token;
      getline(temp_ss, temp_token, ',');
      if (temp_token.size() != 0 && temp_token.at(0) != 'I') {
        std::stringstream _val(temp_token);
        double temp;
        _val >> temp;
        iris_vec.push_back(temp);
      }
    }
  }
  for (int i = 0; i < 150; i++) {
    for (int j = 0; j < 4; j++) {
      iris_mat(i, j) = iris_vec.at(4 * i + j);
    }
  }
  return iris_mat;
}

MatrixXd load_mnist() {
  std::ifstream mnist_data("mnist2500_X.txt");
  MatrixXd mnist_mat(2500, 784);
  std::vector<double> mnist_vec;
  double temp = 0;
  while (mnist_data >> temp) {
    mnist_vec.push_back(temp);
  }

  for (int i = 0; i < 2500; i++) {
    for (int j = 0; j < 784; j++) {
      mnist_mat(i, j) = mnist_vec.at(784 * i + j);
    }
  }
  return mnist_mat;
}