# The nnPU positive weight

Why does the built-in run weigh its revealed mules by one minus the class prior
(`loss.positive_weight = "balanced"`) rather than by the prior, as textbook nnPU does? This
note records the run that collapsed, the simulation that explained it, and how to test the
choice again.

## The run that collapsed

The first full run on the CUDA host trained with textbook nnPU: the positive risk weighted
by the class prior, 0.001 ([reference runs](reference-run.md), run 1). The unlabelled
accounts pushed the scores down about 500 times harder than the 20 revealed mules pulled
them up, and scoring every account near zero costs the objective only the prior. Within one
epoch the loss settled at 0.0010 with every score near 1.7e-7; after 30 epochs (3.3 hours)
the validation proxy AP was 0.017 at a proxy prevalence of 0.0055, and the test proxy ROC
AUC 0.73.

The balanced weight is imbalanced nnPU (Su, Chen and Xu, IJCAI 2021) with a balanced target
prior of 0.5, up to a constant factor. A 40-step run on the graph with it reached
validation proxy AP 0.063 (ROC AUC 0.90) and test proxy AP 0.050 (ROC AUC 0.87), and it
became the built-in setting. Run 2 trained with it reached a ground-truth audit ROC AUC of
0.782; run 3, which adds the pool counts and the slot sum, 0.931.

## The simulation

Does the weight alone explain the collapse? The study simulated a problem with the
dataset's proportions and the repository's loss (`model.loss.NonNegativePULoss`), with no
graph and no ground truth:

- 16 features, the positives shifted by 2 along 4 of them;
- 20 labelled positives, and a label-blind marginal of 20,000 accounts in which positives
  sit at the true prevalence, 0.00073;
- a validation proxy of 11 labelled positives against 2,000 negatives, and a test set of
  300 positives and 300,000 negatives (the class prior, 0.001);
- a small MLP trained on batches of 16 positives drawn with replacement and 48 marginal
  accounts, as the trainer's batches are, for up to 30 epochs of 100 steps, keeping the
  epoch with the best validation proxy AP and stopping after 6 without improvement;
- a run counts as collapsed when its labelled positives score below 0.05 on average: it
  scores even the positives it trains on near zero, the constant scorer that costs the
  textbook objective only the prior.

`mule diagnose nnpu-simulation` runs it (`diagnostics/nnpu_simulation.py`) for the positive
weights 0.001 (the prior), 0.1, 0.5 and 0.999 (balanced) over the seeds 1 to 5. The study's
own runs (`nnpu_sim/grid.py`, three seeds of each weight) printed their results without
keeping them, so the numbers and the figure here are the module's, run offline at its
defaults for this note:

![The nnPU positive weight, simulated](figures/nnpu_simulation.png)

| Positive weight | Collapsed seeds | Test ROC AUC, mean (range) | Test AP, mean | Labelled positives' mean score | Kept epoch, mean |
|---|---|---|---|---|---|
| 0.001 (the prior) | 5 of 5 | 0.379 (0.327 to 0.427) | 0.0008 | 0.0048 | 1.2 |
| 0.1 | 0 of 5 | 0.719 (0.533 to 0.872) | 0.0102 | 0.387 | 5.2 |
| 0.5 | 0 of 5 | 0.859 (0.807 to 0.884) | 0.0296 | 0.904 | 2.2 |
| 0.999 (balanced) | 0 of 5 | 0.865 (0.820 to 0.882) | 0.0288 | 0.956 | 2.0 |

With the prior as its weight, every seed collapses within an epoch or two and ranks worse
than chance, and its non-negative correction never fires. From a weight of 0.5 up, every
seed learns, and the balanced weight ranks best on average, at a test AP of about 30
times the prevalence. So the weight
alone reproduces the collapse on a problem of the dataset's proportions, and the balanced
weight is the one to keep.

## Testing it again

The simulation is a model of the problem, not the problem: its positives are one shifted
Gaussian, and its labelled positives are a uniform draw of them, so the hidden positives
in its marginal are like the labelled ones. The graph reveals the loud mules instead, and
its hidden mules are quieter than the revealed. Its test
on the graph is the `prior_weight` control variant (`loss.positive_weight = "prior"`), which
the control experiments train over the seeds and compare with the baseline on the
validation audit (`python scripts/run_experiments.py`).

## Not carried over

- **The logistic surrogate** (`nnpu_sim/sim.py`), which replaced the loss's sigmoid
  surrogate with a softplus one to see whether the collapse was the surrogate's: the
  repository's loss has one surrogate, and the weight alone reproduces the collapse.
- **The per-epoch trajectories** (`nnpu_sim/traj.py`), which printed the loss, the
  corrected steps and the scores of three settings epoch by epoch: a one-off look at how
  fast the collapse happens, which the kept epochs above summarise.
