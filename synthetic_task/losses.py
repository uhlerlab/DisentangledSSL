import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import softplus
import torch.nn.functional as F


# Batched estimate for KL divergence between any two distributions given their pdf
def batched_kldiv(p1, p2, z1, z2):
    kl12 = p1.log_prob(z1) - p2.log_prob(z1)
    kl21 = p2.log_prob(z2) - p1.log_prob(z2)
    skl = (kl12 + kl21).mean() / 2.
    return skl
        

# Symmetrized Kullback-Leibler divergence between two isotrophic gaussians
def kl_divergence(mu_p, mu_q):
    diff = mu_q - mu_p
    squared_diff = torch.sum(diff * diff, dim=-1)  # Sum along the last dimension (assuming mu_p and mu_q are batched)    
    return squared_diff.mean()  # Take the mean of the KL divergence across the batch and extract the scalar value

def kl_vmf(loc_p, loc_q):
    loc_p = loc_p / loc_p.norm(dim=-1, keepdim=True)
    loc_q = loc_q / loc_q.norm(dim=-1, keepdim=True)
    return -(loc_p * loc_q).sum(-1).mean()

def ortho_loss(z1, zs, norm=True, temp=0.1):
    z1 = F.normalize(z1, dim=-1)
    zs = F.normalize(zs, dim=-1)
    if norm:
        return torch.norm(torch.matmul(z1.T, zs)) # yes (type1)
    else:
        raise NotImplementedError('Please set norm=True')

def ortho_loss_focal(z1, zs):
    assert z1.shape == zs.shape
    z1 = F.normalize(z1, dim=-1)
    zs = F.normalize(zs, dim=-1)
    return torch.matmul(z1, zs.T).diag().mean()


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.ones_like(mask)
        logits_mask[:batch_size, :batch_size] = 0
        logits_mask[batch_size:, batch_size:] = 0
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        with torch.no_grad():
            logits_mask_x = torch.ones_like(mask)
            logits_mask_x[:batch_size, batch_size:] = 0
            logits_mask_x[batch_size:, :batch_size] = 0
            exp_logits_x = torch.exp(logits) * logits_mask_x
            log_prob_x = logits - torch.log(exp_logits_x.sum(1, keepdim=True))
            mask_x = torch.zeros_like(mask)
            mask_x.diagonal().fill_(1)
            mean_log_prob_pos_x = (mask_x * log_prob_x).sum(1) / mask_x.sum(1)
            loss_x = - (self.temperature / self.base_temperature) * mean_log_prob_pos_x
            loss_x, loss_y = loss_x.view(anchor_count, batch_size).mean(1)

        return loss, loss_x, loss_y
    
