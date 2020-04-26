#include "tsne.h"
#include <iostream>
#include <chrono>

pair<MatrixXd, MatrixXd> Hbeta(const Ref<const MatrixXd> &row,
                               const double beta = 1.0) {
  MatrixXd _P = (-row * beta);
  MatrixXd P = _P.array().exp();
  auto sum_P = P.sum();
  MatrixXd H = log(sum_P) + (beta * row.cwiseProduct(P)).array();
  P = P / sum_P;
  return std::make_pair(H, P);
}

MatrixXd PCA(const MatrixXd &X, const int out_dims) {
  if (out_dims < 1) {
    return X;
  }
  BDCSVD<MatrixXd> svd_result(X, Eigen::ComputeThinU | Eigen::ComputeThinV);
  return svd_result.matrixU().leftCols(out_dims) *
         svd_result.singularValues().head(out_dims).asDiagonal() *
         svd_result.matrixV().leftCols(out_dims).transpose();
}

MatrixXd x2p(const MatrixXd &X, const double tol, const double perplexity) {
  auto n = X.rows();
  MatrixXd sum_X = X.cwiseProduct(X).rowwise().sum();
  MatrixXd D =
      ((-2 * X * X.transpose()) + sum_X.replicate(1, X.rows())).transpose() +
      sum_X.replicate(1, X.rows());
  MatrixXd P = MatrixXd::Zero(n, n);
  MatrixXd beta = MatrixXd::Ones(n, 1);
  auto logU = log(perplexity);

  #pragma omp parallel for
  for (int i = 0; i < n; i++) {
    auto beta_min = std::numeric_limits<double>::min();
    auto beta_max = std::numeric_limits<double>::max();
    MatrixXd Di(1, D.cols() - 1);
    for (int idx = 0, Didx = 0; idx < Di.cols(); idx++, Didx++) {
      if (Didx == i) {
        Didx += 1;
      }
      Di(0, idx) = D(i, Didx);
    }
    auto HP = Hbeta(Di, beta(i, 0));
    MatrixXd Hdiff = HP.first.array() - logU;
    auto tries = 0;
    while ((Hdiff.cwiseAbs().array() > tol).all() && tries++ < 50) {
      if ((Hdiff.array() > 0).all()) {
        beta_min = beta(i, 0);
        if (beta_max == std::numeric_limits<double>::max() ||
            beta_max == std::numeric_limits<double>::min()) {
          beta(i, 0) = (beta(i, 0) * 2);
        } else {
          beta(i, 0) = (beta(i, 0) + beta_max) / 2;
        }
      } else {
        beta_max = beta(i, 0);
        if (beta_max == std::numeric_limits<double>::max() ||
            beta_max == std::numeric_limits<double>::min()) {
          beta(i, 0) = (beta(i, 0) / 2);
        } else {
          beta(i, 0) = (beta(i, 0) + beta_min) / 2;
        }
      }
      HP = Hbeta(Di, beta(i, 0));
      Hdiff = HP.first.array() - logU;
    }
    for (int idx = 0, Didx = 0; idx < Di.cols(); idx++, Didx++) {
      if (Didx == i) {
        Didx += 1;
      }
      P(i, Didx) = HP.second(0, idx);
    }
  }
  return P;
}

MatrixXd tSNE(const MatrixXd &X, const int out_dims, const int init_dims,
              const int perplexity, const int max_iter = 1000,
              const double init_momentum = 0.5,
              const double final_momentum = 0.8) {
  auto X_pca = PCA(X, init_dims);
  auto n = X_pca.rows();

  auto eta = 500.0;
  auto min_gain = 0.01;

  MatrixXd Y = MatrixXd::Random(n, out_dims);
  MatrixXd dY = MatrixXd::Zero(n, out_dims);
  MatrixXd iY = MatrixXd::Zero(n, out_dims);
  MatrixXd gains = MatrixXd::Ones(n, out_dims);

  // Computing P-values
  std::cout << "# Computing P-values ..." << std::endl;
  auto P = x2p(X, 1e-5, perplexity);
  P = P + P.transpose().eval();
  P = 4.0 * P / P.sum();
  P = P.cwiseMax(1e-12).eval();

  auto start = std::chrono::high_resolution_clock::now();
  for (int iter = 0; iter < max_iter; iter++) {
    auto sum_Y = Y.cwiseProduct(Y).rowwise().sum();
    MatrixXd num = -2.0 * Y * Y.transpose().eval(); // n x n
    num = (1.0 / (((num + sum_Y.replicate(1, n)).transpose().eval() +
                   sum_Y.replicate(1, n))
                      .array() +
                  1.0))
              .matrix();
    num.diagonal().array() = 0.0;

    MatrixXd Q = num / num.sum();
    Q = Q.cwiseMax(1e-12).eval();

    auto PQ = P - Q; // n x n

    // PQ : n x n
    // num : n x n
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
      dY.row(i) = (PQ.col(i).cwiseProduct(num.col(i)))
                      .replicate(1, out_dims)
                      .cwiseProduct((Y.row(i).replicate(n, 1) - Y))
                      .colwise()
                      .sum();
    }

    auto momentum = final_momentum;
    if (iter < 20) {
      momentum = init_momentum;
    }

    gains =
        ((gains.array() + 0.2) *
             ((dY.array() > 0.0) != (iY.array() > 0.0)).cast<double>().array() +
         (gains.array() * 0.8) *
             ((dY.array() > 0.0) == (iY.array() > 0.0)).cast<double>().array())
            .matrix();

    (gains.array() > min_gain).select(gains, min_gain);
    iY = momentum * iY - eta * (gains.cwiseProduct(dY));
    Y = Y + iY;
    Y = Y - Y.colwise().mean().eval().replicate(n, 1);
    if ((iter + 1) % 100 == 0) {
      auto c = (P.array() * (P.cwiseQuotient(Q).array().log())).sum();
      auto stop = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
      std::cout << "# Iter: " << iter + 1 << ", err: " << c << ", time: " << duration.count() << "s" << std::endl;
      start = std::chrono::high_resolution_clock::now();
    }
    if (iter == 100)
      P = P / 4;
  }

  return Y;
}

void print_matrix_size(const MatrixXd &mat) {
  std::cout << mat.rows() << " " << mat.cols() << std::endl;
}