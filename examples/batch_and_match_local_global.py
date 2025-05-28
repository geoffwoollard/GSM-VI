import torch
from scipy.linalg import sqrtm 


def bam_update(samples, grads, mu0, S0, reg):
    """
    PyTorch implementation of BaM update step.

    Args:
        samples (Tensor): shape (B, D) — batch of samples.
        grads (Tensor): shape (B, D) — score evaluations for each sample.
        mu0 (Tensor): shape (D,) — current mean estimate.
        S0 (Tensor): shape (D, D) — current covariance estimate.
        reg (float): regularization parameter λ > 0.

    Returns:
        mu (Tensor): updated mean (D,)
        S (Tensor): updated covariance matrix (D, D)
    """
    B, D = samples.shape
    xbar = samples.mean(dim=0)
    xdiff = samples - xbar
    C = torch.mean(xdiff.unsqueeze(2) @ xdiff.unsqueeze(1), dim=0)

    gbar = grads.mean(dim=0)
    gdiff = grads - gbar
    G = torch.mean(gdiff.unsqueeze(2) @ gdiff.unsqueeze(1), dim=0)

    U = reg * G + (reg / (1 + reg)) * gbar.unsqueeze(1) @ gbar.unsqueeze(0)
    V = S0 + reg * C + (reg / (1 + reg)) * (mu0 - xbar).unsqueeze(1) @ (mu0 - xbar).unsqueeze(0)

    I = torch.eye(D, dtype=V.dtype, device=V.device)
    mat = I + 4 * U @ V

    sqrt_mat = torch.from_numpy(sqrtm(mat.cpu().numpy())).real

    S = 2 * torch.linalg.solve((I + sqrt_mat).T, V.T).T
    mu = (1 / (1 + reg)) * mu0 + (reg / (1 + reg)) * (S @ gbar + xbar)

    return mu, S


def check_goodness(cov: torch.Tensor) -> bool:
    """
    Check if the covariance matrix is valid (i.e., symmetric positive definite).
    This is done by attempting a Cholesky decomposition and checking for NaNs.

    Args:
        cov (torch.Tensor): A square (D x D) covariance matrix.

    Returns:
        bool: True if the matrix is valid, False otherwise.
    """
    try:
        L = torch.linalg.cholesky(cov)
        return not torch.isnan(L).any()
    except RuntimeError:
        return False
    

def simulator(mu_x, sigma_x, sigma_y, n_samples):
    '''
    x ~  N(mu_x,  sigma_x**2)
    y | x ~ N(x,   sigma_y**2)
    '''
    x = torch.distributions.Normal(mu_x, sigma_x).sample((n_samples,))
    y = torch.distributions.Normal(x, sigma_y).sample()
    return x, y


def posterior_log_prob(x, mu_x, sigma_x, sigma_y, y):
    """
    Compute log p(x | y) for the Gaussian–Gaussian model

        x ~  N(mu_x,  sigma_x**2)
        y | x ~ N(x,   sigma_y**2)

    Parameters
    ----------
    x, mu_x, sigma_x, sigma_y, y : torch.Tensor
        Any shape, will be broadcast automatically.

    Returns
    -------
    torch.Tensor
        log p(x | y) with the broadcasted shape of the inputs.
    """
    tau_x = 1.0 / sigma_x.pow(2)
    tau_y = 1.0 / sigma_y.pow(2)

    var_post = 1.0 / (tau_x + tau_y)               
    mu_post  = var_post * (tau_x * mu_x + tau_y * y)  

    posterior = torch.distributions.Normal(mu_post, var_post.sqrt())
    return posterior.log_prob(x)


def score(z, sigma_y_gt, y_observed, n_mcmc_samples):
    log_sigma_x_estimate = z
    log_sigma_x_estimate.requires_grad = True  
    sigma_x_estimate = torch.exp(log_sigma_x_estimate)
    mu_x_estimate = torch.zeros(len(sigma_x_estimate))  # Assuming mu_x is zero for simplicity

    log_prob = 0
    for y in y_observed:
        x_hidden_sample_thorugh, _ = simulator(mu_x_estimate, sigma_x_estimate, sigma_y_gt, n_samples=n_mcmc_samples)
        log_prob += posterior_log_prob(x_hidden_sample_thorugh, mu_x_estimate, sigma_x_estimate, sigma_y_gt, y).sum()

    log_prob.backward()

    return log_sigma_x_estimate.grad


def batch_and_match_vi(T, B, mu_0, Sigma_0, y_observed, sigma_y_gt, n_mcmc_samples, jitter):

    D = mu_0.shape[0]
    mu_t = mu_0.clone()
    Sigma_t = Sigma_0.clone()

    for t in range(T):

        samples = torch.distributions.MultivariateNormal(mu_t, Sigma_t).sample((B,))

        g_z = torch.zeros(B, D) # (B, D)
        for idx in range(B):
            z_b = samples[idx]
            g_b = score(z_b, sigma_y_gt, y_observed, n_mcmc_samples)
            g_z[idx] = g_b

        lambda_t = T / (t + 1)
        mean_new, cov_new = bam_update(samples, g_z, mu_t, Sigma_t, lambda_t)
        cov_new += torch.eye(D) * jitter
        cov_new = (cov_new + cov_new.T)/2.
        is_good = check_goodness(cov_new)
        if is_good:
            mu_t, Sigma_t = mean_new, cov_new
        else:
            print("Bad update for covariance matrix. Revert")
        if t % 1 == 0:
            print(f"Iteration {t}:")
            print("mean:", mu_t)
            print("cov:", Sigma_t)
            print("Goodness check:", is_good)

    return mu_t, Sigma_t


def main():
    torch.manual_seed(0)
    n_samples = 100
    mu_x_gt = torch.tensor(0.0)
    sigma_x_gt = torch.tensor(1.0)
    sigma_y_gt = torch.tensor(0.01)
    _, y_observed = simulator(mu_x_gt, sigma_x_gt, sigma_y_gt, n_samples)
    d=1
    mu_T, Sigma_T = batch_and_match_vi(T=100, 
                                       B=100, 
                                       mu_0=torch.zeros(d) - 0.1, 
                                       Sigma_0=torch.eye(d) * 1.1,
                                       y_observed=y_observed,
                                       sigma_y_gt=sigma_y_gt,
                                       n_mcmc_samples=300,
                                       jitter=1e-6
                                       )
    print("True mean:", sigma_x_gt)
    print("Final mean: \n log_sigma_x", mu_T)
    print(" sigma_x", mu_T.exp())
if __name__ == "__main__":
    main()
