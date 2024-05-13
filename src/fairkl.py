import torch


def fairkl(feat, labels, bias_labels, temperature=1.0, kld=1.):
    # feat must be normalized!
    bsz = feat.shape[0]

    if labels.shape[0] != bsz:
        raise ValueError('Num of labels does not match num of features')
    if bias_labels.shape[0] != bsz:
        raise ValueError('Num of bias_labels does not match num of features')

    similarity = torch.div(
        torch.matmul(feat, feat.T),
        temperature
    )

    labels = labels.view(-1, 1)
    positive_mask = torch.eq(labels, labels.T)

    bias_labels = bias_labels.view(-1, 1)
    aligned_mask = torch.eq(bias_labels, bias_labels.T)
    conflicting_mask = ~aligned_mask

    pos_conflicting = positive_mask * conflicting_mask
    conflicting_sim = similarity * pos_conflicting
    mu_conflicting = conflicting_sim.sum() / max(pos_conflicting.sum(), 1)
    
    pos_aligned = positive_mask * aligned_mask
    aligned_sim = similarity * pos_aligned
    mu_aligned = aligned_sim.sum() / max(pos_aligned.sum(), 1)

    neg_aligned = (~positive_mask) * aligned_mask
    neg_aligned_sim = similarity * neg_aligned
    mu_neg_aligned = neg_aligned_sim.sum() / max(neg_aligned.sum(), 1)

    neg_conflicting = (~positive_mask) * conflicting_mask
    neg_conflicting_sim = similarity * neg_conflicting
    mu_neg_conflicting = neg_conflicting_sim.sum() / max(neg_conflicting.sum(), 1)

    if mu_conflicting > 1 or mu_conflicting < -1:
        print("mu_conflicting", mu_conflicting.item())

    if mu_aligned > 1 or mu_aligned < -1: 
        print("mu_aligned", mu_aligned.item())

    mu_loss = torch.pow(mu_conflicting - mu_aligned, 2)
    mu_loss += torch.pow(mu_neg_aligned - mu_neg_conflicting, 2)
  
    var_aligned = torch.std(aligned_sim)
    var_conflicting = torch.std(conflicting_sim)
    kld_loss = torch.pow(var_aligned - var_conflicting, 2)

    var_neg_aligned = torch.std(neg_aligned_sim)
    var_neg_conflicting = torch.std(neg_conflicting_sim)
    kld_loss += torch.pow(var_neg_aligned - var_neg_conflicting, 2)

    if torch.isnan(mu_loss):
        print('mu_conflicting:', mu_conflicting)
        print('mu_aligned:', mu_aligned)

    return mu_loss + kld*kld_loss - mu_conflicting + mu_neg_aligned