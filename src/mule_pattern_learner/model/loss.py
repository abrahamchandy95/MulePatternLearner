"""The non-negative PU loss that training minimises."""

from typing import override

import torch
from torch import Tensor
from torch.nn import Module


class NonNegativePULoss(Module):
    """
    Non-negative PU loss for positive-unlabeled binary classification
    (Kiryo et al., 2017).

    Standard cross-entropy is wrong here: the unlabeled set is a MIXTURE of true
    negatives and undiscovered positives (hidden mules), so labelling all
    unlabeled as negative teaches the model to suppress the very signal we want.
    PU learning instead estimates the negative risk indirectly from positives
    and unlabeled, using the known class prior pi.

    With surrogate loss l(z) = sigmoid(-z) (non-increasing), per-sample:
        l_pos_i = l(+f_i) = sigmoid(-f_i)     # loss if i treated as POSITIVE
        l_neg_i = l(-f_i) = sigmoid(+f_i)     # loss if i treated as NEGATIVE

    Risks (n_p = #positives, n_u = #unlabeled in the batch):
        R_p^+   = (1/n_p) * sum_{i in P} l_pos_i           # positives, as positive
        R_p^-   = (1/n_p) * sum_{i in P} l_neg_i           # positives, as negative
        R_u^-   = (1/n_u) * sum_{i in U} l_neg_i           # unlabeled, as negative

    With example weights w each risk is the weighted mean over its group instead:
    n_p and n_u become the groups' total weights and each l_i is multiplied by w_i. Unit
    weights, the default, give the risks above exactly.

        positive_risk = pi * R_p^+
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
        prior: pi = P(y = 1), the assumed TRUE fraction of positives in the
            population (the known simulation mule rate; NOT the revealed-label
            rate, which masking makes artificially low). Must be in (0, 1).
        beta: lower bound the estimated negative risk is clamped against. The
            paper fixes beta = 0.
        gamma: scale on the negative-risk gradient when the correction fires.
            The paper fixes gamma = 1.
        positive_weight: weight on the positive_risk term ONLY. Defaults to
            prior, which reproduces the textbook nnPU objective exactly. Under
            extreme imbalance, optimization may underweight positive examples:
            with prior 0.001 the unlabeled push on the scores is 1 / (2 * prior)
            times the positives' pull, and the cheapest constant scorer, every
            account near zero, costs only the prior, so descent drives all
            scores there. Weight w is imbalanced nnPU (Su, Chen and Xu, IJCAI
            2021) with target prior w / (w + 1 - prior), up to a constant factor
            (exactly, for beta = 0 and gamma = 1): 1 - prior is its balanced
            target prior of 0.5, under which every constant scorer costs
            1 - prior. The scores are then ranking scores, not P(y = 1). No
            weight guarantees ranking quality. The negative-risk correction
            still uses prior. Select this setting on observed validation labels
            and keep hidden truth sealed. Must be in (0, 1).

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
        # Default to prior so the objective is byte-for-byte the textbook nnPU
        # unless the caller deliberately opts into reweighting.
        weight = prior if positive_weight is None else positive_weight
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
        unit = torch.ones_like(logits) if weights is None else weights.to(logits.dtype)
        positive = (targets == 1).to(logits.dtype) * unit  # [N] w_i at positives
        unlabeled = (targets == 0).to(logits.dtype) * unit  # [N] w_i at unlabeled

        # guard against empty groups in a batch (avoid divide-by-zero): a group that
        # weighs nothing adds nothing, so dividing it by 1 leaves it 0
        n_positive = _total(positive)
        n_unlabeled = _total(unlabeled)

        l_pos = self._surrogate_pos(logits)  # [N]
        l_neg = self._surrogate_neg(logits)  # [N]

        # positive_risk = positive_weight * (1/n_p) * sum_{P} l(+f)
        # positive_weight defaults to prior (textbook nnPU); raise it to keep the
        # positives from being weighted into irrelevance under extreme imbalance.
        positive_risk = self._positive_weight * torch.sum(positive * l_pos) / n_positive

        # negative_risk = (1/n_u) sum_{U} l(-f)  -  pi * (1/n_p) sum_{P} l(-f)
        # NOTE: this term keeps the TRUE prior, not positive_weight -- it is the
        # unbiased estimate of the negatives-as-negative risk and must use pi.
        negative_risk = (
            torch.sum(unlabeled * l_neg) / n_unlabeled
            - self._prior * torch.sum(positive * l_neg) / n_positive
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


def _total(weights: Tensor) -> Tensor:
    """The sum of weights, or 1 when it is 0. With unit weights the sum is a count, so this
    equals clamping it at 1."""
    total = weights.sum()
    return torch.where(total > 0, total, torch.ones_like(total))
