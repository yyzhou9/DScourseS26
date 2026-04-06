import numpy as np
from scipy.optimize import minimize
import subprocess

# ============================================================
# Q4: Generate Data
# ============================================================
np.random.seed(100)

N = 100_000
K = 10

# X: N x K matrix, first column is 1s, rest are standard normal
X = np.random.randn(N, K)
X[:, 0] = 1.0  # first column of ones (intercept)

# epsilon ~ N(0, sigma^2), sigma = 0.5
sigma = 0.5
eps = np.random.randn(N) * sigma

# True beta
beta_true = np.array([1.5, -1.0, -0.25, 0.75, 3.5, -2.0, 0.5, 1.0, 1.25, 2.0])

# Generate Y = X @ beta + eps
Y = X @ beta_true + eps

print("=" * 60)
print("Q4: Data generated successfully.")
print(f"  X shape: {X.shape}, Y shape: {Y.shape}")
print(f"  True beta: {beta_true}")

# ============================================================
# Q5: OLS Closed-Form: beta_hat = (X'X)^{-1} X'Y
# ============================================================
XtX = X.T @ X
XtY = X.T @ Y
beta_ols_closed = np.linalg.solve(XtX, XtY)

print("\n" + "=" * 60)
print("Q5: OLS Closed-Form Solution")
print(f"  beta_hat_OLS: {np.round(beta_ols_closed, 6)}")
print(f"  True beta   : {beta_true}")
print(f"  Max abs diff: {np.max(np.abs(beta_ols_closed - beta_true)):.6f}")

# ============================================================
# Q6: OLS via Gradient Descent
# ============================================================
# OLS loss: L(b) = ||Y - Xb||^2
# Gradient: dL/db = -2 X'(Y - Xb)
# Update:   b <- b - lr * gradient

learning_rate = 0.0000003
beta_gd = np.zeros(K)
n_iters = 10_000

for i in range(n_iters):
    residuals = Y - X @ beta_gd
    gradient = -X.T @ residuals  # gradient of (1/2)*RSS, without the 2
    beta_gd = beta_gd - learning_rate * gradient

print("\n" + "=" * 60)
print(f"Q6: OLS via Gradient Descent ({n_iters} iterations, lr={learning_rate})")
print(f"  beta_hat_GD : {np.round(beta_gd, 6)}")
print(f"  True beta   : {beta_true}")
print(f"  Max abs diff: {np.max(np.abs(beta_gd - beta_true)):.6f}")

# ============================================================
# Q7: OLS via scipy L-BFGS-B and Nelder-Mead
# ============================================================
def ols_objective(beta, Y, X):
    residuals = Y - X @ beta
    return 0.5 * np.sum(residuals ** 2)

def ols_gradient(beta, Y, X):
    residuals = Y - X @ beta
    return -X.T @ residuals

beta_init = np.zeros(K)

# L-BFGS-B
res_lbfgs = minimize(
    ols_objective,
    beta_init,
    args=(Y, X),
    method='L-BFGS-B',
    jac=ols_gradient
)
beta_lbfgs = res_lbfgs.x

# Nelder-Mead (no gradient)
res_nm = minimize(
    ols_objective,
    beta_init,
    args=(Y, X),
    method='Nelder-Mead',
    options={'maxiter': 100_000, 'xatol': 1e-8, 'fatol': 1e-8}
)
beta_nm = res_nm.x

print("\n" + "=" * 60)
print("Q7: OLS via Optimization")
print(f"  L-BFGS-B beta_hat : {np.round(beta_lbfgs, 6)}")
print(f"  Nelder-Mead beta_hat: {np.round(beta_nm, 6)}")
print(f"  True beta         : {beta_true}")
print(f"  L-BFGS-B max abs diff  : {np.max(np.abs(beta_lbfgs - beta_true)):.6f}")
print(f"  Nelder-Mead max abs diff: {np.max(np.abs(beta_nm - beta_true)):.6f}")
print(f"  L-BFGS-B converged: {res_lbfgs.success}")
print(f"  Nelder-Mead converged: {res_nm.success}")

# ============================================================
# Q8: MLE via L-BFGS-B
# ============================================================
# Normal MLE: log-likelihood = -N/2 * log(2*pi*sig^2) - 1/(2*sig^2) * ||Y - Xb||^2
# Minimize negative log-likelihood
# theta = [beta (K), sigma (1)]

def mle_neg_loglik(theta, Y, X):
    beta = theta[:K]
    sig = theta[K]
    if sig <= 0:
        return 1e15
    n = len(Y)
    residuals = Y - X @ beta
    nll = (n / 2) * np.log(2 * np.pi * sig**2) + np.sum(residuals**2) / (2 * sig**2)
    return nll

def mle_gradient(theta, Y, X):
    beta = theta[:K]
    sig = theta[K]
    n = len(Y)
    residuals = Y - X @ beta
    grad = np.zeros(K + 1)
    grad[:K] = -X.T @ residuals / sig**2
    grad[K] = n / sig - np.sum(residuals**2) / sig**3
    return grad

theta_init = np.append(np.zeros(K), 1.0)

res_mle = minimize(
    mle_neg_loglik,
    theta_init,
    args=(Y, X),
    method='L-BFGS-B',
    jac=mle_gradient,
    bounds=[(None, None)] * K + [(1e-6, None)]
)
beta_mle = res_mle.x[:K]
sigma_mle = res_mle.x[K]

print("\n" + "=" * 60)
print("Q8: MLE via L-BFGS-B")
print(f"  beta_hat_MLE : {np.round(beta_mle, 6)}")
print(f"  sigma_hat_MLE: {sigma_mle:.6f}  (true sigma = {sigma})")
print(f"  True beta    : {beta_true}")
print(f"  Max abs diff : {np.max(np.abs(beta_mle - beta_true)):.6f}")
print(f"  MLE converged: {res_mle.success}")

# ============================================================
# Q9: OLS the easy way via statsmodels (equivalent to lm())
# ============================================================
import statsmodels.api as sm

# Note: X already has a column of 1s. We use it directly without adding a constant.
model = sm.OLS(Y, X)
results = model.fit()

print("\n" + "=" * 60)
print("Q9: OLS via statsmodels (equivalent to lm(Y ~ X - 1))")
print(results.summary())

# Export regression table to LaTeX
latex_table = results.summary().as_latex()
with open('/home/claude/PS8_Zhou_regtable.tex', 'w') as f:
    f.write(latex_table)

print("\nRegression table saved to PS8_Zhou_regtable.tex")

# ============================================================
# Summary comparison table
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY: Comparison of beta estimates across methods")
print(f"{'Coef':>5} {'True':>8} {'Closed':>10} {'GD':>10} {'LBFGS':>10} {'NM':>10} {'MLE':>10} {'OLS':>10}")
for j in range(K):
    print(f"  b{j:<2} {beta_true[j]:>8.4f} {beta_ols_closed[j]:>10.4f} {beta_gd[j]:>10.4f} "
          f"{beta_lbfgs[j]:>10.4f} {beta_nm[j]:>10.4f} {beta_mle[j]:>10.4f} {results.params[j]:>10.4f}")
