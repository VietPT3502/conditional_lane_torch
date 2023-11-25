import torch
from torch import nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
import numpy as np
import math
import random
def _neg_loss(pred, gt, channel_weights=None):
    ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
      Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred,
                                               2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    if channel_weights is None:
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
    else:
        pos_loss_sum = 0
        neg_loss_sum = 0
        for i in range(len(channel_weights)):
            p = pos_loss[:, i, :, :].sum() * channel_weights[i]
            n = neg_loss[:, i, :, :].sum() * channel_weights[i]
            pos_loss_sum += p
            neg_loss_sum += n
        pos_loss = pos_loss_sum
        neg_loss = neg_loss_sum
    if num_pos > 2:
        loss = loss - (pos_loss + neg_loss) / num_pos
    else:
        loss = loss - (pos_loss + neg_loss) / 256
        loss = torch.tensor(0, dtype=torch.float32).to(pred.device)
    return loss


class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target, weights_list=None):
        return self.neg_loss(out, target, weights_list)


class RegL1KpLoss(nn.Module):

    def __init__(self):
        super(RegL1KpLoss, self).__init__()

    def forward(self, output, target, mask):
        loss = F.l1_loss(output * mask, target * mask, size_average=False)
        mask = mask.bool().float()
        loss = loss / (mask.sum() + 1e-4)
        return loss


class CondLaneLoss(torch.nn.Module):

    def __init__(self, weights):
        """
        Args:
            weights is a dict which sets the weight of the loss
            eg. {hm_weight: 1, kp_weight: 1, ins_weight: 1}
        """
        super(CondLaneLoss, self).__init__()
        self.crit = FocalLoss()
        self.crit_kp = RegL1KpLoss()
        self.crit_ce = nn.CrossEntropyLoss()

        hm_weight = 1.
        kps_weight = 0.4
        row_weight = 1.0
        range_weight = 1.0

        self.hm_weight = weights[
            'hm_weight'] if 'hm_weight' in weights else hm_weight
        self.kps_weight = weights[
            'kps_weight'] if 'kps_weight' in weights else kps_weight
        self.row_weight = weights[
            'row_weight'] if 'row_weight' in weights else row_weight
        self.range_weight = weights[
            'range_weight'] if 'range_weight' in weights else range_weight

    def forward(self, output, **kwargs):
        hm, kps, mask, lane_range = output[:4]
        hm_loss, kps_loss, row_loss, range_loss = 0, 0, 0, 0
        hm = torch.clamp(hm.sigmoid_(), min=1e-4, max=1 - 1e-4)

        if self.hm_weight > 0:
            hm_loss += self.crit(hm, kwargs['gt_hm'])

        if self.kps_weight > 0:
            kps_loss += self.crit_kp(kps, kwargs['gt_reg'],
                                     kwargs['gt_reg_mask'])

        if self.row_weight > 0:
            mask_softmax = F.softmax(mask, dim=3)
            pos = compute_locations2(
                mask_softmax.size(), device=mask_softmax.device)
            
            row_pos = torch.sum(pos * mask_softmax, dim=3) + 0.5
            #print(row_pos*kwargs['gt_row_masks'], kwargs['gt_rows']*kwargs['gt_row_masks'])
            row_loss += self.crit_kp(row_pos, kwargs['gt_rows'],
                                     kwargs['gt_row_masks'])
            #print(row_loss)

        if self.range_weight > 0:
            range_loss = self.crit_ce(lane_range, kwargs['gt_ranges'])

        # Only non-zero losses are valid, otherwise multi-GPU training will report an error
        losses = {}
        loss = 0.
        if self.hm_weight:
            losses['hm_loss'] = self.hm_weight * hm_loss
        if self.kps_weight:
            losses['kps_loss'] = self.kps_weight * kps_loss
        if self.row_weight > 0:
            losses['row_loss'] = self.row_weight * row_loss
        if self.range_weight > 0:
            losses['range_loss'] = self.range_weight * range_loss

        for key, value in losses.items():
            loss += value
        losses['loss'] = loss
        losses['loss_stats'] = losses
        return losses
    
def compute_locations2(shape, device):
    pos = torch.arange(0, shape[-1], step=1, dtype=torch.float32, device=device)
    pos = pos.reshape((1, 1, 1, -1))
    pos = pos.repeat(shape[0], shape[1], shape[2], 1)
    return pos