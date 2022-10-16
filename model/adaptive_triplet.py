"""
@file: adaptive_triplet.py
@time: 2022/09/26
"""
import numpy as np
import torch
import ot
import torch.nn.functional as F


def wasserstein_distance(x, y):
    """ Calculate Wasserstein distance between two regions.
    x.shape=[m,d], y.shape=[n,d]
    """
    off_diag = np.ones((x.shape[0], y.shape[0]))
    indices = np.where(off_diag)
    send_idx = torch.LongTensor(indices[0]).cuda()
    rec_idx = torch.LongTensor(indices[1]).cuda()
    senders = x[send_idx]
    receivers = y[rec_idx]
    # pair-wise matching cost
    similarity = js_div(senders, receivers, get_softmax=True).sum(dim=-1)
    d_cpu = similarity.view(x.shape[0], y.shape[0]).detach().cpu().numpy()
    # calculate optimal transport cost through python optimal transport library
    # This should be faster than using sklearn linear programming api
    p = np.ones(x.shape[0]) / x.shape[0]
    q = np.ones(y.shape[0]) / y.shape[0]
    return ot.emd2(p, q, d_cpu)


def js_div(p_output, q_output, get_softmax=True):
    """
    Function that measures JS divergence between target and output logit:
    Empirically this is better than KL, since it's between 0 and 1.
    """
    if get_softmax:
        p_output = F.softmax(p_output, dim=-1)
        q_output = F.softmax(q_output, dim=-1)
    log_mean_output = ((p_output + q_output) / 2).log()
    return (F.kl_div(log_mean_output, p_output, reduction='none') + F.kl_div(log_mean_output, q_output,
                                                                             reduction='none')) / 2


def adaptive_triplet_loss(anchor, positive, negative, positive_patterns, negative_patterns):
    """
        The proposed adaptive triplet loss function.
        batch_size = 1 always hold.
        Since the wasserstein distance will be very small for similar regions
        We can use a large margin without compromise the embedding quality (e.g. 50 - 100)
        The L1 in land use can be sometimes very low (e.g., 0.46 in Singapore), but we don't do such cherry-picking.
    """
    margin = 100 * wasserstein_distance(positive_patterns, negative_patterns)
    return triplet_loss(anchor, positive, negative, margin=margin)


def triplet_loss(anchor, positive, negative, margin=20):
    """
        For fair comparison, we implement the original triplet loss function, in case that some tricks in pytorch
        implementation of triplet loss affect the performance.
    """
    positive_distance = (anchor - positive).abs().sum(dim=-1)
    negative_distance = (anchor - negative).abs().sum(dim=-1)
    loss = torch.max(positive_distance - negative_distance + margin, torch.zeros((1)).cuda())
    return loss