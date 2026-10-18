"""The non-negative PU loss that training minimises."""

from typing import override

import torch
from torch import Tensor
from torch.nn import Module


class NonNegativePULoss(Module):
    """
    Non-negative PU loss for positive-unlabeled binary classification
    (Kiryo et al., 2017), with balanced weighting (imbalanced nnPU with a balanced
    target, Su, Chen and Xu, IJCAI 2021), as docs/design.md ("Training and
    selection") gives it.

    Standard cross-entropy is wrong here: the unlabeled set is a MIXTURE of true
    negatives and undiscovered positives (hidden mules), so labelling all
    unlabeled as negative teaches the model to suppress the very signal we want.
    PU learning instead estimates the negative risk indirectly from positives
    and unlabeled, using a class prior pi.

    With surrogate loss l(z) = sigmoid(-z) (non-increasing), per-sample:
        l_pos_i = l(+f_i) = 1 - sigmoid(f_i)  # loss if i treated as POSITIVE
        l_neg_i = l(-f_i) = sigmoid(+f_i)     # loss if i treated as NEGATIVE

    Risks (n_p = #positives, n_u = #unlabeled in the batch), with example weights w:
        R_p^+   = (1/n_p) * sum_{i in P} w_i * l_pos_i     # positives, as positive
        R_p^-   = (1/n_p) * sum_{i in P} l_neg_i           # positives, as negative
        R_u^-   = (1/n_u) * sum_{i in U} w_i * l_neg_i     # unlabeled, as negative

    Each sum is divided by its group's count in the batch, n_p or n_u. The positive
    risk keeps each mule's weight of 1 over its active Mondays, so every mule counts
    once however long it ran; the unlabeled risk keeps each draw's weight, so each
    Monday's draw stands for that Monday's scorable population. The correction term
    R_p^- reads every positive at unit weight, because a hidden mule enters the
    unlabeled draw once per active Monday, so the positives it subtracts must count
    the same way. data.examples divides each label's weights by their mean over all
    the training examples, once, so a weight is an example's share of its group's
    influence, and unit weights, the default, give the textbook risks exactly.
    Dividing by the batch's total weight of the group instead would cancel the
    weights' scale in every batch: a lone positive of small weight would fill its
    batch's whole positive risk.

        positive_risk = positive_weight * R_p^+
        negative_risk = R_u^-  -  pi * R_p^-     # unbiased estimate of the
                                                 # negatives-as-negative risk

    Objective:
        uPU:   R = positive_risk + negative_risk
        nnPU:  if negative_risk >= -beta:  R = positive_risk + negative_risk
               else (risk went negative -> overfitting):
                       minimize  gamma * (-negative_risk)
                       i.e. push the negative risk back up toward 0.

    The loss returned for backprop follows the nnPU rule; the always-unclamped
    value is returned beside it for monitoring (the trainer logs its mean and
    the steps where the correction fired).

    Args:
        prior: pi, the share of positives among the scored accounts that the
            loss assumes. The share of hidden active mules among the unlabeled
            is unknown, as in a bank, so training tries a grid of 1, 3 and 10
            times the identified active share of the training world and keeps
            what the selection score prefers; it is never read from the answer
            key. Must be in (0, 1).
        beta: lower bound the estimated negative risk is clamped against. The
            paper fixes beta = 0.
        gamma: scale on the negative-risk gradient when the correction fires.
            The paper fixes gamma = 1.
        positive_weight: weight on the positive_risk term ONLY. Defaults to
            1 - prior, the balanced weighting. With the textbook weight, prior,
            the unlabeled push on the scores is 1 / (2 * prior) times the
            positives' pull at prior 0.001, and the cheapest constant scorer,
            every account near zero, costs only the prior, so descent drives
            all scores there and the loss settles at the prior. Weight w is
            imbalanced nnPU with target prior w / (w + 1 - prior), up to a
            constant factor (exactly, for beta = 0 and gamma = 1): 1 - prior
            is its balanced target prior of 0.5, under which every constant
            scorer costs 1 - prior. The scores are then ranking scores, not
            P(y = 1). No weight guarantees ranking quality. The negative-risk
            correction still uses prior. Must be in (0, 1).

    Convention: targets t use +1 for revealed positives and 0 for unlabeled
    (matching pu_label in this project). f are raw logits, shape [N], and the optional
    weights w are non-negative, shape [N].
    """

    _prior: float
    _beta: float
    _gamma: float
    _positive_weight: float

    def __init__(
        self,
        prior: float,
        beta: float = 0.0,
        gamma: float = 1.0,
        positive_weight: float | None = None,
    ) -> None:
        super().__init__()
        if not (0.0 < prior < 1.0):
            raise ValueError("prior (class prior pi) must be in (0, 1).")
        if gamma <= 0.0:
            raise ValueError("gamma must be positive.")
        # Default to the balanced weight, 1 - prior; prior gives the textbook nnPU.
        weight = 1 - prior if positive_weight is None else positive_weight
        if not (0.0 < weight < 1.0):
            raise ValueError("positive_weight must be in (0, 1).")
        self._prior = prior
        self._beta = beta
        self._gamma = gamma
        self._positive_weight = weight

    @staticmethod
    def _surrogate_pos(logits: Tensor) -> Tensor:
        # l(+f) = sigmoid(-f) = 1 - sigmoid(f): small when f is large (confident positive).
        # Computed from sigmoid(f) so that a badly scored positive (very negative f) keeps
        # its gradient: in float32 sigmoid(-f) rounds to 1 below f of about -17, and its
        # gradient becomes exactly 0. The rounding moves to confident positives instead.
        return 1 - torch.sigmoid(logits)

    @staticmethod
    def _surrogate_neg(logits: Tensor) -> Tensor:
        # l(-f) = sigmoid(+f): small when f is large/negative (confident negative)
        return torch.sigmoid(logits)

    @override
    def forward(
        self, logits: Tensor, targets: Tensor, weights: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        # logits: [N] raw scores f_i ; targets: [N] in {0 unlabeled, 1 positive} ;
        # weights: [N] example weights w_i, 1 each when not given
        is_positive = (targets == 1).to(logits.dtype)  # [N] 1.0 at positives
        is_unlabeled = (targets == 0).to(logits.dtype)  # [N] 1.0 at unlabeled

        # guard against empty groups in a batch (avoid divide-by-zero)
        n_positive = torch.clamp(is_positive.sum(), min=1.0)
        n_unlabeled = torch.clamp(is_unlabeled.sum(), min=1.0)

        unit = torch.ones_like(logits) if weights is None else weights.to(logits.dtype)
        positive = is_positive * unit  # [N] w_i at positives
        unlabeled = is_unlabeled * unit  # [N] w_i at unlabeled

        l_pos = self._surrogate_pos(logits)  # [N]
        l_neg = self._surrogate_neg(logits)  # [N]

        # positive_risk = positive_weight * (1/n_p) * sum_{P} w * l(+f)
        # positive_weight defaults to 1 - prior (balanced), which keeps the
        # positives from being weighted into irrelevance under extreme imbalance.
        positive_risk = self._positive_weight * torch.sum(positive * l_pos) / n_positive

        # negative_risk = (1/n_u) sum_{U} w * l(-f)  -  pi * (1/n_p) sum_{P} l(-f)
        # The correction keeps the prior, not positive_weight, since it is the
        # unbiased estimate of the negatives-as-negative risk, and reads every
        # positive at unit weight.
        negative_risk = (
            torch.sum(unlabeled * l_neg) / n_unlabeled
            - self._prior * torch.sum(is_positive * l_neg) / n_positive
        )

        objective = positive_risk + negative_risk

        # nnPU correction: if the estimated negative risk dips below -beta, the
        # unbiased objective is overfitting; replace the backprop target with the
        # gradient-ascending term gamma * (-negative_risk) to push it back up.
        # torch.where keeps the branch on the device (no host synchronization);
        # the unselected branch receives an exact zero gradient, so values and
        # gradients equal the former Python branch.
        train_loss = torch.where(
            negative_risk < -self._beta, self._gamma * (-negative_risk), objective
        )

        return train_loss, objective
