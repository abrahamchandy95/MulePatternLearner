from pathlib import Path

import pandas as pd


def main() -> None:
    masks = sorted(
        Path("data/masks").glob("pu_labels_*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not masks:
        print("no parquet found in data/masks")
        return
    frame = pd.read_parquet(masks[0])
    print(f"parquet: {masks[0].name}\n")

    # split codes: 0=train, 1=val, 2=test
    split_names = {0: "train", 1: "val", 2: "test"}
    print(f"{'split':>6} {'accounts':>10} {'true_mules':>12} {'revealed':>10} {'hidden':>8}")
    for code, name in split_names.items():
        sub = frame[frame["split"] == code]
        n = len(sub)
        true_mules = int((sub["true_label"] == 1).sum())
        revealed = int((sub["bucket"] == 1).sum())
        hidden = int((sub["bucket"] == 2).sum())
        print(f"{name:>6} {n:>10} {true_mules:>12} {revealed:>10} {hidden:>8}")

    val = frame[frame["split"] == 1]
    val_mules = int((val["true_label"] == 1).sum())
    print()
    if val_mules == 0:
        print("PROBLEM: the val split has ZERO true mules. Ranking metrics are")
        print("undefined on it, and train.py would fail the same way. The split")
        print("needs to guarantee positives land in val (stratified split).")
    else:
        print(f"OK: val has {val_mules} true mules, so AP/AUC are computable on the")
        print("full val set. The smoke test only failed because it took the first 64")
        print("accounts, which missed them. Fixing the smoke test is enough.")


if __name__ == "__main__":
    main()
