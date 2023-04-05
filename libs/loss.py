import torch
import torch.nn.functional as F

def mse(recon_x, x):
    loss = torch.sum(torch.square(recon_x - x), dim=1)
    loss = torch.sum(loss) / x.size(0)

    return loss

def mse_samples(recon_x, x):
    losses = torch.sum(torch.square(recon_x - x), dim=1)

    return losses


def apre(recon_x, x, alpha=1, beta=1):
    recon_mu, recon_sigma = recon_x.chunk(2, dim=1)
    recon_sigma = F.softplus(recon_sigma)
    loss = alpha * torch.sum(torch.square(recon_mu - x) / (recon_sigma + 1e-5), dim=1)
    loss += beta * torch.sum(torch.log((recon_sigma + 1e-5)), dim=1)
    loss = torch.sum(loss) / x.size(0)

    return loss

def apre_samples(recon_x, x, alpha=1, beta=1):
    recon_mu, recon_sigma = recon_x.chunk(2, dim=1)
    recon_sigma = F.softplus(recon_sigma)
    losses = alpha * torch.sum(torch.square(recon_mu - x) / (recon_sigma + 1e-5), dim=1)
    losses += beta * torch.sum(torch.log((recon_sigma + 1e-5)), dim=1)

    return losses

