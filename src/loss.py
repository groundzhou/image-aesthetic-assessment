import torch
import torch.nn as nn


class EMDLoss(nn.Module):
    """EMDLoss class
    """
    def __init__(self):
        super(EMDLoss, self).__init__()

    def forward(self, p_true: torch.Tensor, p_pred: torch.Tensor):
        assert p_true.shape == p_pred.shape, 'Length of the two distribution must be the same'
        cdf_target = torch.cumsum(p_true, dim=1)  # cdf for values [1, 2, ..., 10]
        cdf_estimate = torch.cumsum(p_pred, dim=1)  # cdf for values [1, 2, ..., 10]
        cdf_diff = cdf_estimate - cdf_target
        samplewise_emd = torch.sqrt(torch.mean(torch.pow(torch.abs(cdf_diff), 2), dim=1))
        return samplewise_emd.mean()

