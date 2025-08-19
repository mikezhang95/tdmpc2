import torch
import torch.nn.functional as F


def soft_ce(pred, target, cfg):
    """Computes the cross entropy loss between predictions and soft targets."""
    target = two_hot(target, cfg)
    if cfg.num_bins <= 1: # num_bins <=1 downgrade to mse
        return (target - pred)**2
    pred = F.log_softmax(pred, dim=-1)
    return -(target * pred).sum(-1, keepdim=True)


def log_std(x, low, dif):
    return low + 0.5 * dif * (torch.tanh(x) + 1)


def _gaussian_residual(eps, log_std):
    return -0.5 * eps.pow(2) - log_std


def _gaussian_logprob(residual):
    log2pi = 1.8378770351409912
    return residual - 0.5 * log2pi


def gaussian_logprob(eps, log_std, size=None):
    """Compute Gaussian log probability."""
    residual = _gaussian_residual(eps, log_std).sum(-1, keepdim=True)
    if size is None:
        size = eps.shape[-1]
    return _gaussian_logprob(residual) * size


def _squash(pi):
    return torch.log(F.relu(1 - pi.pow(2)) + 1e-6)


def squash(mu, pi, log_pi):
    """Apply squashing function."""
    mu = torch.tanh(mu)
    pi = torch.tanh(pi)
    log_pi -= _squash(pi).sum(-1, keepdim=True)
    return mu, pi, log_pi


def symlog(x):
    """
    Symmetric logarithmic function.
    Adapted from https://github.com/danijar/dreamerv3.
    """
    return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x):
    """
    Symmetric exponential function.
    Adapted from https://github.com/danijar/dreamerv3.
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def two_hot(x, cfg):
    """Converts a batch of scalars to soft two-hot encoded targets for discrete regression."""
    if cfg.num_bins == 0:
        return x
    elif cfg.num_bins == 1:
        return symlog(x)
    x = torch.clamp(symlog(x), cfg.vmin, cfg.vmax).squeeze(1)
    bin_idx = torch.floor((x - cfg.vmin) / cfg.bin_size)
    bin_offset = ((x - cfg.vmin) / cfg.bin_size - bin_idx).unsqueeze(-1)
    soft_two_hot = torch.zeros(x.shape[0], cfg.num_bins, device=x.device, dtype=x.dtype)
    bin_idx = bin_idx.long()
    soft_two_hot = soft_two_hot.scatter(1, bin_idx.unsqueeze(1), 1 - bin_offset)
    soft_two_hot = soft_two_hot.scatter(1, (bin_idx.unsqueeze(1) + 1) % cfg.num_bins, bin_offset)
    return soft_two_hot


def two_hot_inv(x, cfg):
    """Converts a batch of soft two-hot encoded vectors to scalars."""
    if cfg.num_bins == 0:
        return x
    elif cfg.num_bins == 1:
        return symexp(x)
    dreg_bins = torch.linspace(cfg.vmin, cfg.vmax, cfg.num_bins, device=x.device, dtype=x.dtype)
    x = F.softmax(x, dim=-1)
    x = torch.sum(x * dreg_bins, dim=-1, keepdim=True)
    return symexp(x)


def gumbel_softmax_sample(p, temperature=1.0, dim=0):
    logits = p.log()
    # Generate Gumbel noise
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / temperature  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)
    return y_soft.argmax(-1)

def softmax_distillation_loss(q_student: torch.Tensor, q_teacher: torch.Tensor, temperature: float = 1.0):
    """
    Args:
        q_student: [B, A] tensor Q-values predicted by student.
        q_teacher: [B, A] tensor Q-values predicted by teacher.
        temperature: float temperature for softening the distributions.
    Returns:
        loss: scalar listwise distillation loss.
    """
    # Soften Q-values
    q_t = q_teacher / temperature
    q_s = q_student / temperature

    # Convert to log probabilities
    log_probs_student = F.log_softmax(q_s, dim=-1)        # [B, A]
    probs_teacher = F.softmax(q_t, dim=-1).detach()       # [B, A], stop gradient from teacher

    # KL divergence per sample
    loss = F.kl_div(log_probs_student, probs_teacher, reduction='batchmean')
    return loss

def vae_loss(recon, raw, mu, logvar):
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon, raw, reduction='mean')
    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

