#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/SVD"
#include <limits>
#include <utility>

using namespace Eigen;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::pair;

pair<double, MatrixXd> Hbeta(const Ref<const MatrixXd> &row,
                               const double beta);
MatrixXd tSNE(const MatrixXd &X, const int out_dims, const int init_dims,
              const int perplexity, const int max_iter,
              const double init_momentum,
              const double final_momentum);
MatrixXd PCA(const MatrixXd &X, const int out_dims);
MatrixXd x2p(const MatrixXd &X, const double tol, const double perplexity);
void print_matrix_size(const MatrixXd &mat);
