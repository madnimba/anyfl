# Thin wrapper: runs four swap strategies (optimal, round_robin, random_clusters, random_per_sample)
# for pred and gt cluster modes. All logic in attack_core and run_attack.
from __future__ import annotations

from run_attack import load_data, run_strategies
from attack_core import EPOCHS

DATASETS = ["HAR", "MUSHROOM"]
FRACTION_TRAIN = 0.85
TRAIN_SAMPLES = 5000
TEST_SAMPLES = 1000


def main():
    for dataset_name in DATASETS:
        print("=====================================================")
        print(f"Dataset: {dataset_name}")
        XA_train, XB_train, Y_train, XA_test, XB_test, Y_test = load_data(
            dataset_name, TRAIN_SAMPLES, TEST_SAMPLES
        )
        run_strategies(
            dataset_name,
            XA_train, XB_train, Y_train,
            XA_test, XB_test, Y_test,
            epochs=EPOCHS,
            use_signature=True,
        )


if __name__ == "__main__":
    main()
