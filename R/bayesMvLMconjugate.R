#' Gibbs sampler for Conjugate Bayesian Multivariate Linear Models
#'
#' @param Y [matrix] \eqn{n \times q} of response variables
#' @param X [matrix] \eqn{n \times p} of predictors
#' @param mu_B [matrix] \eqn{p \times q} prior mean for \eqn{\beta}
#' @param V_B [matrix] \eqn{p \times p} prior row covariance for \eqn{\beta}
#' @param nu [double] prior parameter for \eqn{\Sigma}
#' @param Psi [matrix] prior parameter for \eqn{\Sigma}
#' @param n_iter [integer] iteration number for Gibbs sampler
#' @param burn_in [integer] number of burn-in iteration
#'
#' @return B_samples [array] of posterior sample for \eqn{\beta}
#' @return Sigma_samples [array] of posterior samples for \eqn{\Sigma}
#'
#' @importFrom mniw rMNorm riwish
#' @importFrom utils txtProgressBar setTxtProgressBar
#'
#' @examples
#' ## Generate data
#' n <- 100
#' p <- 3
#' q <- 2
#' Y <- matrix(rnorm(n*q), nrow = n, ncol = q)
#' X <- matrix(rnorm(n*p), nrow = n, ncol = p)
#'
#' ## Prior parameters
#' mu_B <- matrix(0, p, q)
#' V_B <- diag(10, p)
#' nu <- 3
#' Psi <- diag(q)
#'
#' ## Samples from posteriors
#' n_iter <- 1000
#' burn_in <- 500
#' set.seed(1234)
#' samples <- spBPS::bayesMvLMconjugate(Y, X, mu_B, V_B, nu, Psi, n_iter, burn_in)
#'
#' @export
bayesMvLMconjugate <- function(Y, X, mu_B, V_B, nu, Psi, n_iter = 1000, burn_in = 500) {
  n <- nrow(Y)
  q <- ncol(Y)
  p <- ncol(X)

  # Initialize matrices to store samples
  B_samples <- array(0, dim = c(p, q, n_iter))
  Sigma_samples <- array(0, dim = c(q, q, n_iter))

  # Initial values
  B <- matrix(0, p, q)
  Sigma <- diag(q)

  pb <- txtProgressBar(0, n_iter, style = 3)

  # Gibbs sampler iterations
  for (iter in 1:n_iter) {
    # Sample B given Sigma and data
    V_n <- solve(solve(V_B) + t(X) %*% X)
    B_n <- V_n %*% (solve(V_B) %*% mu_B + t(X) %*% Y)
    B <- mniw::rMNorm(1, B_n, V_n, Sigma)

    # Sample Sigma given B and data
    S_n <- Psi + t(Y - X %*% B) %*% (Y - X %*% B)
    Sigma <- mniw::riwish(1, nu = nu + n, Psi = S_n)

    # Store samples
    B_samples[, , iter] <- B
    Sigma_samples[, , iter] <- Sigma

    setTxtProgressBar(pb, iter)
  }

  # Remove burn-in samples
  B_samples <- B_samples[, , (burn_in + 1):n_iter]
  Sigma_samples <- Sigma_samples[, , (burn_in + 1):n_iter]

  list(B_samples = B_samples, Sigma_samples = Sigma_samples)
}

#' Predictive sampler for Conjugate Bayesian Multivariate Linear Models
#'
#' @param X_new [matrix] \eqn{n_new \times p} of predictors for new data points
#' @param B_samples [array] of posterior sample for \eqn{\beta}
#' @param Sigma_samples [array] of posterior samples for \eqn{\Sigma}
#'
#' @return Y_pred [matrix] of posterior mean for response matrix Y predictions
#' @return Y_pred_samples [array] of posterior predictive sample for response matrix Y
#'
#' @importFrom mniw rMNorm
#' @importFrom utils txtProgressBar setTxtProgressBar
#'
#' @examples
#' ## Generate data
#' n <- 100
#' p <- 3
#' q <- 2
#' Y <- matrix(rnorm(n*q), nrow = n, ncol = q)
#' X <- matrix(rnorm(n*p), nrow = n, ncol = p)
#'
#' ## Prior parameters
#' mu_B <- matrix(0, p, q)
#' V_B <- diag(10, p)
#' nu <- 3
#' Psi <- diag(q)
#'
#' ## Samples from posteriors
#' n_iter <- 1000
#' burn_in <- 500
#' set.seed(1234)
#' samples <- spBPS::bayesMvLMconjugate(Y, X, mu_B, V_B, nu, Psi, n_iter, burn_in)
#'
#' ## Extract posterior samples
#' B_samples <- samples$B_samples
#' Sigma_samples <- samples$Sigma_samples
#'
#' ## Samples from predictive posterior (based posterior samples)
#' m <- 50
#' X_new <- matrix(rnorm(m*p), nrow = m, ncol = p)
#' pred <- spBPS::pred_bayesMvLMconjugate(X_new, B_samples, Sigma_samples)
#'
#' @export
pred_bayesMvLMconjugate <- function(X_new, B_samples, Sigma_samples) {
  n_iter <- dim(B_samples)[3]
  n_new <- nrow(X_new)
  q <- dim(Sigma_samples)[1]

  # Initialize matrix to store predictions
  Y_pred_samples <- array(0, dim = c(n_new, q, n_iter))
  Y_pred <- matrix(0, n_new, q)

  # Loop over the posterior samples
  pb <- txtProgressBar(0, n_iter, style = 3)
  for (iter in 1:n_iter) {
    B <- B_samples[, , iter]
    Sigma <- Sigma_samples[, , iter]

    # Predict responses for the new data
    Y_pred_iter <- mniw::rMNorm(n = 1, Lambda = X_new %*% B, SigmaR = diag(n_new), SigmaC = Sigma)

    # Store samples
    Y_pred_samples[, , iter] <- Y_pred_iter

    # Add the predictions to the cumulative sum
    Y_pred <- Y_pred + Y_pred_iter

    setTxtProgressBar(pb, iter)
  }

  # Average the predictions over all posterior samples
  Y_pred <- Y_pred / n_iter

  list(Y_pred_samples = Y_pred_samples, Y_pred = Y_pred)

}
