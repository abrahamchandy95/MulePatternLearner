from collections.abc import Iterator, Sequence
import random


class SeedPool:
    """
    Account-id pools for one split, partitioned by PU label.

    Separates the few revealed positives (pu_label == 1) from the unlabeled
    majority (pu_label == 0), because the nnPU loss needs positives present in
    every batch (its positive_risk term is estimated only from positives). The
    ids come from the caller (read from the graph), so this holds no I/O.

    positives:  ids with pu_label == 1
    unlabeled:  ids with pu_label == 0
    """

    positives: tuple[str, ...]
    unlabeled: tuple[str, ...]

    def __init__(self, positives: Sequence[str], unlabeled: Sequence[str]) -> None:
        self.positives = tuple(positives)
        self.unlabeled = tuple(unlabeled)

    @property
    def num_positives(self) -> int:
        return len(self.positives)

    @property
    def num_unlabeled(self) -> int:
        return len(self.unlabeled)


def epoch_batches(
    pool: SeedPool,
    batch_size: int,
    positives_per_batch: int,
    seed: int,
) -> Iterator[tuple[str, ...]]:
    """
    Yield one epoch of seed-id batches that oversample the rare positives.

    Each batch contains exactly positives_per_batch positive seeds (sampled WITH
    replacement, since there may be fewer unique positives than batches need)
    plus (batch_size - positives_per_batch) unlabeled seeds (sampled WITHOUT
    replacement within the epoch, so the unlabeled set is covered once). The
    number of batches is set so the unlabeled pool is consumed once per epoch:
    n_batches = ceil(num_unlabeled / unlabeled_per_batch).

    This guarantees nnPU's positive_risk always has positives to average over,
    while still traversing the unlabeled data once per epoch. Each yielded tuple
    is the seed_ids argument for backend.make_loader.

    Raises ValueError if positives_per_batch is not in [1, batch_size), or if
    the pool has no positives (nnPU cannot train without them).
    """
    if positives_per_batch < 1:
        raise ValueError("positives_per_batch must be >= 1 for nnPU.")
    if positives_per_batch >= batch_size:
        raise ValueError("positives_per_batch must be < batch_size.")
    if pool.num_positives == 0:
        raise ValueError("seed pool has no positives; nnPU cannot train.")

    rng = random.Random(seed)
    unlabeled_per_batch = batch_size - positives_per_batch

    # shuffle the unlabeled pool once; we consume it in order, one pass per epoch
    unlabeled = list(pool.unlabeled)
    rng.shuffle(unlabeled)

    if pool.num_unlabeled == 0:
        n_batches = 1
    else:
        n_batches = (pool.num_unlabeled + unlabeled_per_batch - 1) // unlabeled_per_batch

    positives = list(pool.positives)

    for b in range(n_batches):
        # positives: sample WITH replacement (few uniques, need them every batch)
        pos_seeds = [positives[rng.randrange(len(positives))] for _ in range(positives_per_batch)]

        # unlabeled: take the next slice of the shuffled pool (without replacement)
        start = b * unlabeled_per_batch
        unl_seeds = unlabeled[start : start + unlabeled_per_batch]

        batch = pos_seeds + unl_seeds
        rng.shuffle(batch)  # interleave so order doesn't encode the label
        yield tuple(batch)
