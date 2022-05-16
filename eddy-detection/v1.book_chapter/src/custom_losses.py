import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    def __init__(self, num_classes, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.num_classes = num_classes
        if weight is None:
            weight = torch.ones(num_classes) / num_classes
        # make sure weights get transferred to GPU if loss_fn.to("cuda") is called
        self.register_buffer("weights", torch.Tensor(weight / sum(weight)))
        assert self.num_classes == len(
            self.weights
        ), f"num_classes {self.num_classes} != len(weights) {len(self.weights)}"

    def forward(self, y_pred, y_true, smooth=1):
        """
        Definition: dice = f1 = harmonic mean of precision and recall
        f1 = 2 * (Precision * Recall) / (Precision + Recall) = 2TP/(2TP + FN + FP)
        Given y_pred, and y_true, TP = |y_pred * y_true|
        FN = y_true - TP (i.e., number of positive samples that weren't predicted as positive)
        FP = y_pred - TP (i.e., number of predicted +ve amples that were NOT positive)
        So 2TP + FN + FP = 2TP + (y_true - TP) + (y_pred - TP) = y_true + y_pred
        And f1 = 2TP/(y_true + y_pred)
        """
        cross_entropy = F.cross_entropy(
            y_pred, y_true, weight=self.weights, reduction="mean"
        )
        # cross_entropy = self.cross_entropy(y_pred, y_true)
        # comment out if your model contains a sigmoid or equivalent activation layer
        y_pred = F.softmax(y_pred, dim=-1)

        # need to make sure that y_pred and y_true are categorical/one-hot encoded
        if y_pred.shape[-1] == 1:
            y_pred = F.one_hot(y_pred, num_classes=self.num_classes)

        if y_true.ndim == 1 or y_true.shape[-1] == 1:
            y_true = F.one_hot(y_true, num_classes=self.num_classes)

        # We need to sum across all dimensions but the last one
        # y_pred.shape = [N, 128, 128, 3] or [N, 128*128, 3]
        # y_true.shape is the same as y_pred.shape
        # if ndim = 4, sum_dims = (0, 1, 2)
        # if ndim = 2, sum_dims = (0)
        sum_dims = tuple(dim for dim in range(0, y_pred.ndim - 1))

        # dice will be a 1xN matrix, where N = num_classes
        # here, we do matrix multiplication to end up at the right shape
        TP_each_class = (y_pred * y_true).sum(sum_dims)
        num_true_each_class = y_true.sum(sum_dims)
        num_pred_true = y_pred.sum(sum_dims)

        # get the dice score for each class
        dice = (2.0 * TP_each_class + smooth) / (
            num_true_each_class + num_pred_true + smooth
        )

        # macro averaging across dice scores for each class
        weighted_dice = (self.weights * dice).sum()

        # convert to loss function; lower is better
        weighted_dice_loss = 1 - weighted_dice.mean()  # macro avg
        dice_ce_loss = cross_entropy + weighted_dice_loss

        return dice_ce_loss
