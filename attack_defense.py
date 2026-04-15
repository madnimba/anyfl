# Thin wrapper: pred vs GT defense suite for tabular (and any) dataset.
# Data loading and split here; all attack/defense logic in attack_core and run_attack.
from __future__ import annotations

import torch

from datasets import get_dataset
from attack_core import (
    set_seed,
    SEED,
    EPOCHS,
    BATCH_SIZE,
    to_XA_XB_Y_from_numpy,
    make_swapped_XA,
    run_defense_suite_once,
    pretty_print_suite,
    pretty_print_compare,
    train_once,
    evaluate,
    make_clean_z_stream,
    DEFENSES,
    DEFENSE_ORDER,
)

DATASETS = ["MUSHROOM"]
TRAIN_SAMPLES = 5000
TEST_SAMPLES = 1000


def main():
    set_seed(SEED)
    for dataset_name in DATASETS:
        print("=====================================================")
        print(f"Dataset: {dataset_name}")
        X_np, y_np = get_dataset(dataset_name)
        XA_all, XB_all, Y_all = to_XA_XB_Y_from_numpy(X_np, y_np)
        N = len(Y_all)
        n_train_req = min(TRAIN_SAMPLES, max(1, N - 1))
        n_test_req = min(TEST_SAMPLES, max(1, N - n_train_req))
        g = torch.Generator().manual_seed(SEED)
        perm = torch.randperm(N, generator=g)
        XA_all = XA_all[perm]
        XB_all = XB_all[perm]
        Y_all = Y_all[perm]
        train_idx = perm[:n_train_req]

        XA_train = XA_all[:n_train_req]
        XB_train = XB_all[:n_train_req]
        Y_train = Y_all[:n_train_req]
        XA_test = XA_all[n_train_req : n_train_req + n_test_req]
        XB_test = XB_all[n_train_req : n_train_req + n_test_req]
        Y_test = Y_all[n_train_req : n_train_req + n_test_req]

        cleanA, cleanB, cleanC, _ = train_once(
            XA_train, XB_train, Y_train, defense_name="none", epochs=EPOCHS,
            dataset_name=dataset_name,
        )
        acc_clean = evaluate(cleanA, cleanB, cleanC, XA_test, XB_test, Y_test)
        print(f"[CLEAN] Accuracy: {acc_clean*100:.2f}%")

        frozen_def = DEFENSES["frozen_ae"]
        frozen_def.reset()
        stream = make_clean_z_stream(cleanA, XA_train, batch=BATCH_SIZE, steps=800)
        frozen_def._ensure(stream[0].shape[1])
        frozen_def.pretrain_on_clean_stream(stream)

        XA_sw_pred, note_pred = make_swapped_XA(
            dataset_name, XA_train, Y_train, mode="pred", select_idx=train_idx
        )
        XA_sw_gt, note_gt = make_swapped_XA(
            dataset_name, XA_train, Y_train, mode="gt"
        )
        print(note_pred)
        print(note_gt)

        print("\n[ATTACK] Running defense suite with PREDICTED clusters...")
        res_pred = run_defense_suite_once(
            dataset_name,
            XA_sw_pred,
            XB_train,
            Y_train,
            XA_test,
            XB_test,
            Y_test,
            epochs=EPOCHS,
            seed=SEED,
            XA_clean_train=XA_train,
        )
        print("[ATTACK] Running defense suite with GROUND-TRUTH clusters...")
        res_gt = run_defense_suite_once(
            dataset_name,
            XA_sw_gt,
            XB_train,
            Y_train,
            XA_test,
            XB_test,
            Y_test,
            epochs=EPOCHS,
            seed=SEED,
            XA_clean_train=XA_train,
        )
        pretty_print_suite("Predicted Clusters", res_pred)
        pretty_print_suite("Ground-Truth Clusters", res_gt)
        pretty_print_compare("Predicted", res_pred, "GroundTruth", res_gt)
        print()


if __name__ == "__main__":
    main()
