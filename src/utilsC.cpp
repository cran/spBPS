#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>
#include <R.h>
#include <Rmath.h>
#include <Rinternals.h>
#include "utilsC.h"

// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;


// UTILITY FUNCTIONS ------------------------------------------------------------------------------


//' Compute the Euclidean distance matrix
//'
//' @param X [matrix] (tipically of \eqn{N} coordindates on \eqn{\mathbb{R}^2} )
//'
//' @return [matrix] distance matrix of the elements of \eqn{X}
//'
//' @examples
//' ## Compute the Distance matrix of dimension (n x n)
//' n <- 100
//' p <- 2
//' X <- matrix(runif(n*p), nrow = n, ncol = p)
//' distance.matrix <- arma_dist(X)
//'
//' @export
// [[Rcpp::export(name = "arma_dist")]]
arma::mat arma_dist(const arma::mat & X){
  int n = X.n_rows;
  arma::mat D(n, n, fill::zeros); // Allocate a matrix of dimension n x n
  for (int i = 0; i < n; i++) {
    for(int k = 0; k < i; k++){
      D(i, k) = sqrt(sum(pow(X.row(i) - X.row(k), 2)));
      D(k, i) = D(i, k);
    }
  }
  return D;
}


//' Build a grid from two vector (i.e. equivalent to \code{expand.grid()} in \code{R})
//'
//' @param x [vector] first vector of numeric elements
//' @param y [vector] second vector of numeric elements
//'
//' @return [matrix] expanded grid of combinations
//'
//' @examples
//' ## Create a matrix from all combination of vectors
//' x <- seq(0, 10, length.out = 100)
//' y <- seq(-1, 1, length.out = 20)
//' grid <- expand_grid_cpp(x = x, y = y)
//'
//' @export
// [[Rcpp::export]]
arma::mat expand_grid_cpp(const arma::vec& x, const arma::vec& y) {
  int n1 = x.size();
  int n2 = y.size();
  int n_combinations = n1 * n2;

  arma::mat result(n_combinations, 2);

  int k = 0;
  for (int j = 0; j < n2; j++) {
    for (int i = 0; i < n1; i++) {
      result(k, 0) = x[i];
      result(k, 1) = y[j];
      k++;
    }
  }

  return result;
}


//' Function to sample integers (index)
//'
//' @param size [integer] dimension of the set to sample
//' @param length [integer] number of elements to sample
//' @param p [vector] sampling probabilities
//'
//' @return [vector] sample of integers
//'
// [[Rcpp::export]]
arma::uvec sample_index(const int& size, const int& length, const arma::vec& p){
  arma::uvec sequence = arma::linspace<arma::uvec>(0, size-1, size);
  arma::uvec out = Rcpp::RcppArmadillo::sample(sequence, length, true, p);
  return out;
}


//' Function to subset data for meta-analysis
//'
//' @param data [list] three elements: first named \eqn{Y}, second named \eqn{X}, third named \eqn{crd}
//' @param K [integer] number of desired subsets
//'
//' @return [list] subsets of data, and the set of indexes
//'
//' @examples
//' ## Create a list of K random subsets given a list with Y, X, and crd
//' n <- 100
//' p <- 3
//' q <- 2
//' X <- matrix(rnorm(n*p), nrow = n, ncol = p)
//' Y <- matrix(rnorm(n*q), nrow = n, ncol = q)
//' crd <- matrix(runif(n*2), nrow = n, ncol = 2)
//' subsets <- subset_data(data = list(Y = Y, X = X, crd = crd), K = 10)
//'
//' @export
// [[Rcpp::export]]
List subset_data(const List& data, int K) {

  // Unpack data and priors
  arma::mat Y = as<arma::mat>(data["Y"]);
  arma::mat X = as<arma::mat>(data["X"]);
  arma::mat crd = as<arma::mat>(data["crd"]);

  // Setting set size
  int n = Y.n_rows;
  int set_size = floor(n / K);

  arma::uvec indices = arma::randperm(n);

  // Subsetting
  List Y_list(K);
  List X_list(K);
  List crd_list(K);
  List sets(K);

  for (int k = 0; k < K; ++k) {

    // set index
    arma::uvec ind_k = indices.subvec( (k * set_size), ((k + 1) * set_size) - 1);

    // subset data
    Y_list[k] = arma::conv_to<arma::mat>::from(Y.rows(ind_k));
    X_list[k] = arma::conv_to<arma::mat>::from(X.rows(ind_k));
    crd_list[k] = arma::conv_to<arma::mat>::from(crd.rows(ind_k));
    sets[k] = ind_k;

  }

  return List::create(Named("Y_list") = Y_list,
                      Named("X_list") = X_list,
                      Named("crd_list") = crd_list,
                      Named("sets") = sets);
}


//' Function to subset data for meta-analysis
//'
//' @param mat [matrix] not-symmetric matrix
//'
//' @return [matrix] symmetric matrix (lower triangular of \code{mat} is used)
//'
//' @examples
//' ## Force matrix to be symmetric (avoiding numerical problems)
//' n <- 4
//' X <- matrix(runif(n*n), nrow = n, ncol = n)
//' X <- forceSymmetry_cpp(mat = X)
//'
//' @export
// [[Rcpp::export]]
arma::mat forceSymmetry_cpp(const arma::mat& mat) {
  // Extract the lower triangular part of the matrix
  arma::mat lower = arma::trimatl(mat);

  // Create a symmetric matrix by copying the lower triangular part to the upper triangular part
  arma::mat symmat = arma::symmatl(lower);

  return symmat;
}






