#include "tsne.cpp"
// #include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main() {
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
            if (temp_token.size() != 0 && temp_token.at(0) != 'i') {
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

    MatrixXd result = tSNE(iris_mat, 2, -1, 50.0);
    
    std::cout << result << std::endl;

    return 0;
}