import torch

def negative_binomial_loss(pred, target, theta, eps=1e-8):
    """
    Negative binomial loss with learnable dispersion parameter theta.

    Args:
        pred: torch.Tensor, predicted mean parameter (mu).
        target: torch.Tensor, observed counts (y).
        theta: torch.Tensor, dispersion parameter (theta), must be positive.
        eps: float, small value for numerical stability.

    Returns:
        torch.Tensor: Scalar loss (mean negative log-likelihood).
    """
    # Ensure stability
    pred = pred.clamp(min=eps)
    theta = theta.clamp(min=eps)

    # Negative binomial log likelihood
    log_prob = (
        torch.lgamma(target + theta)
        - torch.lgamma(theta)
        - torch.lgamma(target + 1)
        + theta * (torch.log(theta + eps) - torch.log(theta + pred + eps))
        + target * (torch.log(pred + eps) - torch.log(theta + pred + eps))
    )

    return -log_prob.mean()
