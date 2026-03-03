#!/usr/bin/env python3
"""
Single entrypoint for VFL attack experiments.
Usage:
  python run_attack.py --dataset MNIST --experiment suite
  python run_attack.py --dataset MUSHROOM --experiment strategies
  python run_attack.py --dataset MNIST --experiment sensitivity --k 1 2 3 5
"""
import argparse
from typing import List, Tuple

import torch
from torchvision import datasets, transforms

from datasets import get_dataset
from attack_core import (
    set_seed,
    SEED,
    EPOCHS,
    BATCH_SIZE,
    load_cluster_info,
    make_swapped_XA,
    build_swapped_variants,
    run_defense_suite_once,
    pretty_print_suite,
    pretty_print_compare,
    print_strategy_table,
    train_once,
    evaluate,
    make_clean_z_stream,
    DEFENSES,
    to_XA_XB_Y_from_numpy,
    _infer_topk_targets,
    generate_cluster_swapped_attack_topk,
)

# Image datasets: use torchvision train/test split. Keys uppercase so --dataset mnist/fashionmnist/cifar10 all work.
IMAGE_DATASETS = {
    "MNIST": datasets.MNIST,
    "FASHIONMNIST": datasets.FashionMNIST,
    "KMNIST": datasets.KMNIST,
    "CIFAR10": datasets.CIFAR10,
}


def vertical_split(img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    w = img.shape[-1]
    mid = w // 2
    return img[:, :, :mid], img[:, :, mid:]


def load_image_data(
    dataset_name: str,
    train_samples: int,
    test_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tf = transforms.ToTensor()
    cls = IMAGE_DATASETS[dataset_name.strip().upper()]
    train_data = cls(".", train=True, download=True, transform=tf)
    test_data = cls(".", train=False, download=True, transform=tf)

    def split(data):
        XA, XB, Y = [], [], []
        for img, label in data:
            a, b = vertical_split(img)
            XA.append(a)
            XB.append(b)
            Y.append(label)
        return torch.stack(XA), torch.stack(XB), torch.tensor(Y)

    XA_tr, XB_tr, Y_tr = split(train_data)
    XA_te, XB_te, Y_te = split(test_data)
    return (
        XA_tr[:train_samples],
        XB_tr[:train_samples],
        Y_tr[:train_samples],
        XA_te[:test_samples],
        XB_te[:test_samples],
        Y_te[:test_samples],
    )


def load_tabular_data(
    dataset_name: str,
    train_samples: int,
    test_samples: int,
    fraction_train: float = 0.85,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # get_dataset accepts e.g. mnist, cifar10, 20ng, mushroom, bank, har, fashionmnist (case-insensitive)
    X_np, y_np = get_dataset(dataset_name.strip())
    XA_all, XB_all, Y_all = to_XA_XB_Y_from_numpy(X_np, y_np)
    N = len(Y_all)
    n_train = min(train_samples, max(1, int(fraction_train * N)))
    n_train = max(1, min(n_train, N - 1))
    n_test = min(test_samples, N - n_train)
    # Use first n_train as train so cluster file order (first N) matches train set
    return (
        XA_all[:n_train],
        XB_all[:n_train],
        Y_all[:n_train],
        XA_all[n_train : n_train + n_test],
        XB_all[n_train : n_train + n_test],
        Y_all[n_train : n_train + n_test],
    )


def load_data(
    dataset_name: str,
    train_samples: int = 5000,
    test_samples: int = 1000,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    d = dataset_name.strip().upper()
    if d in IMAGE_DATASETS:
        return load_image_data(d, train_samples, test_samples)
    return load_tabular_data(d, train_samples, test_samples)


def run_suite(
    dataset_name: str,
    XA_train: torch.Tensor,
    XB_train: torch.Tensor,
    Y_train: torch.Tensor,
    XA_test: torch.Tensor,
    XB_test: torch.Tensor,
    Y_test: torch.Tensor,
    epochs: int = EPOCHS,
    pretrain_frozen_ae: bool = True,
) -> None:
    set_seed(SEED)
    cleanA, cleanB, cleanC, _ = train_once(
        XA_train, XB_train, Y_train, defense_name="none", epochs=epochs
    )
    acc_clean = evaluate(cleanA, cleanB, cleanC, XA_test, XB_test, Y_test)
    print(f"[CLEAN] Accuracy: {acc_clean*100:.2f}%")

    if pretrain_frozen_ae:
        frozen_def = DEFENSES["frozen_ae"]
        frozen_def.reset()
        stream = make_clean_z_stream(cleanA, XA_train, batch=BATCH_SIZE, steps=800)
        frozen_def._ensure(stream[0].shape[1])
        frozen_def.pretrain_on_clean_stream(stream)

    XA_sw_pred, note_pred = make_swapped_XA(
        dataset_name, XA_train, Y_train, mode="pred", n_needed=len(Y_train)
    )
    XA_sw_gt, note_gt = make_swapped_XA(dataset_name, XA_train, Y_train, mode="gt")
    print(note_pred)
    print(note_gt)

    res_pred = run_defense_suite_once(
        dataset_name,
        XA_sw_pred,
        XB_train,
        Y_train,
        XA_test,
        XB_test,
        Y_test,
        epochs=epochs,
        seed=SEED,
        XA_clean_train=XA_train,
    )
    res_gt = run_defense_suite_once(
        dataset_name,
        XA_sw_gt,
        XB_train,
        Y_train,
        XA_test,
        XB_test,
        Y_test,
        epochs=epochs,
        seed=SEED,
        XA_clean_train=XA_train,
    )
    pretty_print_suite("Predicted Clusters", res_pred)
    pretty_print_suite("Ground-Truth Clusters", res_gt)
    pretty_print_compare("Predicted", res_pred, "GroundTruth", res_gt)


def run_sensitivity(
    dataset_name: str,
    XA_train: torch.Tensor,
    XB_train: torch.Tensor,
    Y_train: torch.Tensor,
    XA_test: torch.Tensor,
    XB_test: torch.Tensor,
    Y_test: torch.Tensor,
    k_values: List[int],
    epochs: int = EPOCHS,
) -> None:
    set_seed(SEED)
    cleanA, cleanB, cleanC, _ = train_once(
        XA_train, XB_train, Y_train, defense_name="none", epochs=epochs
    )
    acc_clean = evaluate(cleanA, cleanB, cleanC, XA_test, XB_test, Y_test)
    print(f"Clean accuracy: {acc_clean*100:.2f}%\n")
    print("k  | Attack Accuracy (%) | Drop (ppt)")
    print("--------------------------------------")

    cluster_info = load_cluster_info(dataset_name, len(Y_train))
    if cluster_info is None:
        print("[WARN] No clusters found; run clustering first.")
        return

    for k_val in k_values:
        topk = _infer_topk_targets(dataset_name, XA_train, cluster_info["ids"], k=k_val)
        XA_sw = generate_cluster_swapped_attack_topk(
            XA_train,
            cluster_info["ids"],
            topk_map=topk,
            conf=cluster_info.get("conf"),
            core_q=0.60,
            seed=SEED,
        )
        A, B, C, _ = train_once(
            XA_sw, XB_train, Y_train, defense_name="none", epochs=epochs
        )
        acc_k = evaluate(A, B, C, XA_test, XB_test, Y_test)
        drop = (acc_clean - acc_k) * 100
        print(f"{k_val:<2} | {acc_k*100:>8.2f}%         | -{drop:.2f}")
    print()


def run_strategies(
    dataset_name: str,
    XA_train: torch.Tensor,
    XB_train: torch.Tensor,
    Y_train: torch.Tensor,
    XA_test: torch.Tensor,
    XB_test: torch.Tensor,
    Y_test: torch.Tensor,
    epochs: int = EPOCHS,
    use_signature: bool = False,
) -> None:
    set_seed(SEED)
    cleanA, cleanB, cleanC, _ = train_once(
        XA_train, XB_train, Y_train, defense_name="none", epochs=epochs
    )
    acc_clean = evaluate(cleanA, cleanB, cleanC, XA_test, XB_test, Y_test)
    print(f"[CLEAN] Accuracy: {acc_clean*100:.2f}%")

    pred_groups = pred_conf = None
    cluster_info = load_cluster_info(dataset_name, n_needed=len(Y_train))
    if cluster_info is not None:
        pred_groups = cluster_info["ids"]
        pred_conf = cluster_info.get("conf")

    for mode in ["pred", "gt"]:
        variants = build_swapped_variants(
            dataset_name,
            XA_train,
            Y_train,
            mode=mode,
            pred_groups=pred_groups,
            pred_conf=pred_conf,
            use_signature=use_signature,
        )
        print(f"\n[ATTACK] Swap strategies with mode={mode.upper()} ...")
        results = {}
        for strat_name, (XA_swapped, note) in variants.items():
            print(f"  • {strat_name}: {note[:60]}...")
            acc_by_def = run_defense_suite_once(
                dataset_name,
                XA_swapped,
                XB_train,
                Y_train,
                XA_test,
                XB_test,
                Y_test,
                epochs=epochs,
                seed=SEED,
                XA_clean_train=XA_train,
            )
            results[strat_name] = {k: v["acc"] for k, v in acc_by_def.items()}
        print_strategy_table(
            f"{dataset_name} | mode={mode.upper()} | Accuracy by defense",
            results,
        )
    print()


def main():
    p = argparse.ArgumentParser(description="VFL attack experiments")
    p.add_argument("--dataset", type=str, default="MNIST", help="Dataset name")
    p.add_argument(
        "--experiment",
        type=str,
        choices=["suite", "sensitivity", "strategies"],
        default="suite",
        help="suite=pred vs GT defense table; sensitivity=top-k; strategies=4 swap strategies",
    )
    p.add_argument("--train_samples", type=int, default=5000)
    p.add_argument("--test_samples", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--k", type=int, nargs="+", default=[1, 2, 3, 5, 7], help="k values for sensitivity")
    p.add_argument("--use_signature", action="store_true", help="Use cluster signature in cache (swap_strategies style)")
    args = p.parse_args()

    print(f"Dataset: {args.dataset}")
    XA_train, XB_train, Y_train, XA_test, XB_test, Y_test = load_data(
        args.dataset, args.train_samples, args.test_samples
    )
    print(f"Train: {len(Y_train)}, Test: {len(Y_test)}")

    if args.experiment == "suite":
        run_suite(
            args.dataset,
            XA_train, XB_train, Y_train,
            XA_test, XB_test, Y_test,
            epochs=args.epochs,
        )
    elif args.experiment == "sensitivity":
        run_sensitivity(
            args.dataset,
            XA_train, XB_train, Y_train,
            XA_test, XB_test, Y_test,
            k_values=args.k,
            epochs=args.epochs,
        )
    else:
        run_strategies(
            args.dataset,
            XA_train, XB_train, Y_train,
            XA_test, XB_test, Y_test,
            epochs=args.epochs,
            use_signature=args.use_signature,
        )


if __name__ == "__main__":
    main()
