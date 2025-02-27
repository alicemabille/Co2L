from __future__ import print_function

import torch
import torch.nn as nn

'''
The original source code can be found in
https://github.com/HobbitLong/SupContrast/blob/master/losses.py
'''


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, target_labels=None, reduction='mean'):
        assert target_labels is not None and len(target_labels) > 0, "Target labels should be given as a list of integer"

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

        contrast_count = features.shape[1] # number of augmentations for 1 sample
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        #This splits features into a tuple (t1, t2), where:
        #    t1 has shape (bsz, D)
        #    t2 has shape (bsz, D)
        #So torch.unbind(features, dim=1) returns (t1, t2)

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
            self.temperature) # (zi · zj/τ )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        print("logits shape : ",logits.shape)

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask # exp(zi · zj/τ ) where j!=i and yj = yi
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True)) 
        # log ( exp(zi · zj/τ ) / ∑ k6=i exp(zi · zk/τ ) ) = log(exp(zi · zj/τ )) - log(∑ k6=i exp(zi · zk/τ ))
        # = (zi · zj/τ ) - log(∑ k6=i exp(zi · zk/τ ))
        print("log_prob shape : ",log_prob.shape)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # mask.sum(1) = number of positive sample for current sample xi
        print("mean_log_prob_pos, mean of log-likelihood over positive, shape : ",mean_log_prob_pos.shape)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        curr_class_mask = torch.zeros_like(labels)
        for tc in target_labels:
            curr_class_mask += (labels == tc)
        curr_class_mask = curr_class_mask.view(-1).to(device)
        loss = curr_class_mask * loss.view(anchor_count, batch_size)

        if reduction == 'mean':
            loss = loss.mean()
        elif reduction == 'none':
            loss = loss.mean(0)
        else:
            raise ValueError('loss reduction not supported: {}'.
                             format(reduction))

        return loss, mean_log_prob_pos, mask #TODO : choose what loss values to return for buffer management
