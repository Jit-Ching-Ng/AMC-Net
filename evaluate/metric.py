import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()
        self.smooth = 1e-5

    def forward(self, prediction, ground_truth):
        intersection = torch.sum(prediction * ground_truth)
        union = torch.sum(prediction) + torch.sum(ground_truth)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice
        # loss = F.binary_cross_entropy_with_logits(prediction, ground_truth)
        return loss


def dice_coefficient(prediction, ground_truth):
    intersection = np.logical_and(prediction, ground_truth)
    return 2.0 * intersection.sum() / (prediction.sum() + ground_truth.sum())


def iou(prediction, ground_truth):
    intersection = np.logical_and(prediction, ground_truth)
    union = np.logical_or(prediction, ground_truth)
    return intersection.sum() / union.sum()


def recall(prediction, ground_truth):
    true_positives = np.logical_and(prediction, ground_truth)
    return true_positives.sum() / ground_truth.sum()


def precision(prediction, ground_truth):
    true_positives = np.logical_and(prediction, ground_truth)
    return true_positives.sum() / prediction.sum()


def jaccard_index(prediction, ground_truth):
    intersection = np.logical_and(prediction, ground_truth)
    union = np.logical_or(prediction, ground_truth)
    return intersection.sum() / union.sum()
