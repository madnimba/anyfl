# Thin wrapper: runs top-k sensitivity (and optional full suite) for image datasets.
# All logic lives in attack_core and run_attack.
from __future__ import annotations

from run_attack import load_data, run_sensitivity
from attack_core import EPOCHS

DATASETS = ["MNIST"]
K_VALUES = [1, 2, 3, 5, 7]
TRAIN_SAMPLES = 5000
TEST_SAMPLES = 1000


def main():
    print("\n========================================")
    print(" TOP-K SWAP SENSITIVITY ANALYSIS (NO DEFENSE)")
    print("========================================\n")

    for dataset_name in DATASETS:
        print(f"Dataset = {dataset_name}")
        XA_train, XB_train, Y_train, XA_test, XB_test, Y_test = load_data(
            dataset_name, TRAIN_SAMPLES, TEST_SAMPLES
        )
        run_sensitivity(
            dataset_name,
            XA_train, XB_train, Y_train,
            XA_test, XB_test, Y_test,
            k_values=K_VALUES,
            epochs=EPOCHS,
        )


if __name__ == "__main__":
    main()
