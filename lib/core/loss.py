# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch

class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints
        


class FieldsLoss(nn.Module):
    def __init__(self):
        super(FieldsLoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)

    def forward(self, fields, target, weight):
        batch_size = fields.size(0)
        num_fields = fields.size(1)
        fields_pred = fields.reshape((batch_size, num_fields, -1)).split(1, 1)
        fields_gt = target.reshape((batch_size, num_fields, -1)).split(1, 1)

        loss = 0

        for idx in range(num_fields):
            fld_pred = fields_pred[idx].squeeze()
            fld_gt = fields_gt[idx].squeeze()
            mask = (fld_gt>0).type(torch.cuda.FloatTensor)
            mask = mask.mul(weight[:, int(idx/2)])
            loss += self.criterion(fld_pred.mul(mask), fld_gt.mul(mask))

        return loss / num_fields
