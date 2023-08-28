import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class FFDS(nn.Module):
    def __init__(
        self,
        class_freq,
        groups,
        class_weight: torch.Tensor,
        smoothing_alpha: float = 0.1,
        freq_gamma_min: float = 0,
        freq_gamma_max: float = 3,
        prob_smooth_percentage_alpha: float = 0.9,
        gamma_type="concave",
        reduction: str = "mean",
    ) -> None:
        nn.Module.__init__(self)

        assert prob_smooth_percentage_alpha <= 1 and \
            prob_smooth_percentage_alpha >= 0
        assert gamma_type in ["linear", "convex", "concave"]
        assert freq_gamma_max >= freq_gamma_min

        self.class_freq = class_freq
        self.reduction = reduction
        n_classes = len(class_freq)

        self.class_weight = class_weight
        # label smoothing_alpha
        self.smoothing_alpha = smoothing_alpha
        self.gamma_type = gamma_type

        device = class_weight.device
        self.groups = torch.tensor(groups)
        self.mem_groups_prob = torch.zeros(
            len(groups), dtype=torch.float, device=device
        )
        self.class2groups_mean = torch.zeros(
            len(self.class_freq), dtype=torch.float, device=device
        )
        self.mem_groups_count = torch.zeros(
            len(groups), dtype=torch.float, device=device
        )

        self.prob_smooth_percentage_alpha = prob_smooth_percentage_alpha

        self.freq_gamma_max = freq_gamma_max
        self.freq_gamma_min = freq_gamma_min

        self._calculate_gamma()

        # ls
        self.ls_target = torch.zeros(n_classes, n_classes, device=device)
        self.ls_target.fill_(self.smoothing_alpha / (n_classes - 1))
        self.ls_target.fill_diagonal_(1 - self.smoothing_alpha)

    def forward(self, pred: Tensor, truth: Tensor):
        # prevent numerical instability
        const = -torch.max(pred, dim=1).values
        const = torch.unsqueeze(const, -1)
        probs = F.softmax(pred + const, dim=1)
        EPS = 1e-6
        probs = torch.clamp(probs, EPS, 1.0 - EPS)

        if self.training:
            with torch.no_grad():
                # get ground truth probs
                gt_probs = torch.gather(probs, -1, torch.unsqueeze(truth, -1))
                gt_probs = torch.squeeze(gt_probs)
                batch_freq = torch.bincount(truth,
                                            minlength=self.class_freq.size(0))

                # accumulate per-group probability and count
                if self.groups.size(0) == 1:
                    self.mem_groups_count[0] += batch_freq.sum()
                    self.mem_groups_prob[0] += gt_probs.sum()
                else:
                    self.mem_groups_count[0] += batch_freq[0:self.groups[0]].sum()
                    self.mem_groups_prob[0] += gt_probs[truth < self.groups[0]].sum()

                    for i in range(1, len(self.groups)):
                        # mask for particular group
                        mask = (truth >= self.groups[i - 1]) & \
                            (truth < self.groups[i])
                        self.mem_groups_prob[i] += gt_probs[mask].sum()
                        self.mem_groups_count[i] += batch_freq[
                            self.groups[i - 1]: self.groups[i]].sum()

        # if no group mean yet (at first epoch)
        if (self.class2groups_mean == 0).all():
            loss = F.cross_entropy(pred, truth, reduction="none")
        else:
            loss = self._soft_loss(probs, truth)
            smoothed_probs = self._calculate_prob_smooth(gt_probs, truth)
            loss *= (torch.pow(1 - smoothed_probs, self.gamma[truth]) *
                     self.class_weight[truth])

        if self.reduction == "none":
            return loss
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.sum() / probs.shape[0]
        else:
            raise ValueError("reduction must be 'none', 'sum', or 'mean'")

    def _soft_loss(self, probs: Tensor, truth: Tensor):
        with torch.no_grad():
            true_dist = torch.index_select(self.ls_target, 0, truth)
        return torch.sum(-true_dist * torch.log(probs), -1)

    def _calculate_prob_smooth(self, probs: Tensor, truth: Tensor):
        sqrt_d = torch.sqrt(probs) - torch.sqrt(self.class2groups_mean[truth])
        prob_smooth_percentage = torch.pow(
            self.prob_smooth_percentage_alpha * sqrt_d, 2)

        smooth = prob_smooth_percentage * (probs -
                                           self.class2groups_mean[truth])
        final_probs = probs - smooth
        assert torch.all((final_probs >= 0) & (final_probs <= 1))

        return final_probs

    def _calculate_gamma(self):
        if self.gamma_type == "linear":
            self.gamma = self.freq_gamma_min + \
                (self.freq_gamma_max - self.freq_gamma_min) * \
                (self.class_freq - self.class_freq[-1]) / \
                (self.class_freq[0] - self.class_freq[-1])
        elif self.gamma_type == "concave":
            self.gamma = self.freq_gamma_min + \
                (self.freq_gamma_max - self.freq_gamma_min) * \
                torch.tanh(4 * (self.class_freq - self.class_freq[-1]) /
                           (self.class_freq[0] - self.class_freq[-1]))
        elif self.gamma_type == "convex":
            self.gamma = self.freq_gamma_min + \
                (self.freq_gamma_max - self.freq_gamma_min) * \
                torch.pow((self.class_freq - self.class_freq[-1]) /
                          (self.class_freq[0] - self.class_freq[-1]), 3)
        assert torch.all(self.gamma >= self.freq_gamma_min) and torch.all(
            self.gamma <= self.freq_gamma_max)

    def next_epoch(self):
        with torch.no_grad():
            last_groups_mean = self.mem_groups_prob / self.mem_groups_count

            self.mem_groups_count.zero_()
            self.mem_groups_prob.zero_()

            group_count = 0
            for idx, _ in enumerate(range(len(self.class_freq))):
                if idx == self.groups[group_count]:
                    group_count += 1
                if idx < self.groups[group_count]:
                    self.class2groups_mean[idx] = last_groups_mean[group_count]
