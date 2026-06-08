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
    value is also reported for monitoring.

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
            extreme imbalance (pi ~ 0.003) the textbook weight makes
            positive_risk a negligible fraction of the loss (here ~0.001), so
            the optimizer minimizes the unlabeled term alone by driving every
            logit down -- dragging positives BELOW unlabeled and inverting the
            ranking. Raising positive_weight (e.g. to pi-balanced 0.5, or any
            value >> pi) restores gradient signal to the positives WITHOUT
            touching the prior used in the unbiased negative-risk correction
            below, so that estimator stays unbiased. This separation is the fix:
            it reweights how much positives matter, not the statistics of the
            negatives-as-negative estimate. Must be in (0, 1).

    Convention: targets t use +1 for revealed positives and 0 for unlabeled
    (matching pu_label in this project). f are raw logits, shape [N].
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
        # l(+f) = sigmoid(-f): small when f is large/positive (confident positive)
        return torch.sigmoid(-logits)

    @staticmethod
    def _surrogate_neg(logits: Tensor) -> Tensor:
        # l(-f) = sigmoid(+f): small when f is large/negative (confident negative)
        return torch.sigmoid(logits)

    @override
    def forward(self, logits: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
        # logits: [N] raw scores f_i ; targets: [N] in {0 unlabeled, 1 positive}
        positive = (targets == 1).to(logits.dtype)  # [N] 1.0 at positives
        unlabeled = (targets == 0).to(logits.dtype)  # [N] 1.0 at unlabeled

        # guard against empty groups in a batch (avoid divide-by-zero)
        n_positive = torch.clamp(positive.sum(), min=1.0)
        n_unlabeled = torch.clamp(unlabeled.sum(), min=1.0)

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
        if negative_risk.item() < -self._beta:
            train_loss = self._gamma * (-negative_risk)
        else:
            train_loss = objective

        return train_loss, objective
