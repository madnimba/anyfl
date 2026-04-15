#!/usr/bin/env python3
"""
Single entrypoint for VFL attack experiments.
Usage:
  python run_attack.py --dataset MNIST --experiment suite
  python run_attack.py --dataset MUSHROOM --experiment strategies
  python run_attack.py --dataset MNIST --experiment sensitivity --k 1 2 3 5
"""
import argparse
import os
from datetime import datetime
from typing import List, Optional, Set, Tuple

import torch
from torchvision import datasets, transforms

from datasets import get_dataset
from attack_core import (
    DEVICE,
    set_seed,
    SEED,
    EPOCHS,
    CIFAR10_EPOCHS,
    BATCH_SIZE,
    load_cluster_info,
    make_swapped_XA,
    build_swapped_variants,
    run_defense_suite_once,
    pretty_print_suite,
    pretty_print_compare,
    print_strategy_table,
    train_once,
    train_once_rgar,
    evaluate,
    make_clean_z_stream,
    DEFENSES,
    DEFENSE_ORDER,
    verify_cluster_files,
    to_XA_XB_Y_from_numpy,
    _infer_topk_targets,
    generate_cluster_swapped_attack_topk,
    _reset_everything,
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
    dname = dataset_name.strip().upper()
    if dname == "CIFAR10":
        tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010),
                ),
            ]
        )
    else:
        tf = transforms.ToTensor()
    cls = IMAGE_DATASETS[dname]
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
        XA_train, XB_train, Y_train, defense_name="none", epochs=epochs,
        dataset_name=dataset_name,
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
        XA_train, XB_train, Y_train, defense_name="none", epochs=epochs,
        dataset_name=dataset_name,
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
            XA_sw, XB_train, Y_train, defense_name="none", epochs=epochs,
            dataset_name=dataset_name,
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
        XA_train, XB_train, Y_train, defense_name="none", epochs=epochs,
        dataset_name=dataset_name,
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


def _compare_epochs(dataset_name: str, epochs: int, quick: bool) -> int:
    """CIFAR-10 needs long SGD training for high clean accuracy; MNIST is cheaper."""
    d = dataset_name.strip().upper()
    if d == "CIFAR10":
        if quick:
            # Allow short smoke runs; still need enough SGD steps to see signal.
            return max(8, min(epochs, CIFAR10_EPOCHS))
        if epochs < 100:
            print(
                f"[CIFAR10] Raising training epochs from {epochs} to {CIFAR10_EPOCHS} "
                f"so clean accuracy can approach strong VFL baselines.",
                flush=True,
            )
            return CIFAR10_EPOCHS
        return epochs
    if quick:
        return min(12, max(6, epochs))
    return epochs


def run_compare_rgar(
    dataset_name: str,
    XA_train: torch.Tensor,
    XB_train: torch.Tensor,
    Y_train: torch.Tensor,
    XA_test: torch.Tensor,
    XB_test: torch.Tensor,
    Y_test: torch.Tensor,
    epochs: int,
    out_path: str = "results_rgar_compare.txt",
    quick: bool = False,
    defense_keys: Optional[Set[str]] = None,
) -> None:
    """
    One full cluster-swap attack + legacy DEFENSE_ORDER sweep + RGAR (full + downweight-only).
    Uses predicted clusters when ./clusters/{DATASET}_ids.npy exists; otherwise GT label-swap (with warning).
    """
    set_seed(SEED)
    ep = _compare_epochs(dataset_name, epochs, quick)
    swap_mode = "pred"
    try:
        verify_cluster_files(dataset_name)
    except FileNotFoundError as err:
        print(f"[WARN] {err}\n[WARN] Falling back to ground-truth label swap for this run.")
        swap_mode = "gt"

    cleanA, cleanB, cleanC, _ = train_once(
        XA_train,
        XB_train,
        Y_train,
        defense_name="none",
        epochs=ep,
        batch_size=BATCH_SIZE,
        dataset_name=dataset_name,
    )
    acc_clean = evaluate(cleanA, cleanB, cleanC, XA_test, XB_test, Y_test)
    print(f"[{dataset_name}] CLEAN accuracy: {acc_clean * 100:.2f}%", flush=True)

    frozen_def = DEFENSES["frozen_ae"]
    frozen_def.reset()
    stream = make_clean_z_stream(cleanA, XA_train, batch=BATCH_SIZE, steps=800)
    frozen_def._ensure(stream[0].shape[1])
    frozen_def.pretrain_on_clean_stream(stream)

    XA_sw, note_sw = make_swapped_XA(
        dataset_name,
        XA_train,
        Y_train,
        mode=swap_mode,
        n_needed=len(Y_train),
    )
    print(note_sw, flush=True)

    _reset_everything(SEED)
    legacy = run_defense_suite_once(
        dataset_name,
        XA_sw,
        XB_train,
        Y_train,
        XA_test,
        XB_test,
        Y_test,
        epochs=ep,
        seed=SEED,
        XA_clean_train=XA_train,
        batch_size=BATCH_SIZE,
        defense_keys=defense_keys,
    )

    from server_rgar_defense import RGARConfig

    rgar_cfg = RGARConfig(
        ref_warmup_epochs=5 if quick else 14,
        recon_epochs=100 if quick else 360,
    )

    _reset_everything(SEED)
    Af, Bf, Cf, _, meta_full = train_once_rgar(
        XA_train,
        XA_sw,
        XB_train,
        Y_train,
        dataset_name=dataset_name,
        epochs=ep,
        batch_size=BATCH_SIZE,
        seed=SEED,
        rgar_cfg=rgar_cfg,
        downweight_only=False,
    )
    acc_rgar_full = evaluate(Af, Bf, Cf, XA_test, XB_test, Y_test)

    _reset_everything(SEED)
    Ad, Bd, Cd, _, meta_dw = train_once_rgar(
        XA_train,
        XA_sw,
        XB_train,
        Y_train,
        dataset_name=dataset_name,
        epochs=ep,
        batch_size=BATCH_SIZE,
        seed=SEED,
        rgar_cfg=rgar_cfg,
        downweight_only=True,
    )
    acc_rgar_dw = evaluate(Ad, Bd, Cd, XA_test, XB_test, Y_test)

    def _meta_short(m: dict) -> str:
        return f"rho_a={m.get('rho_a', 0):.2f} attrib_A={m.get('attributed_malicious_a', False)}"

    det_full = float(meta_full.get("attack_detect_rate_pct", 0.0))
    det_dw = float(meta_dw.get("attack_detect_rate_pct", 0.0))

    lines: List[str] = []
    ts = datetime.now().isoformat()
    lines.append(f"# RGAR vs legacy defenses — {ts}")
    dk_note = "all" if defense_keys is None else ",".join(sorted(defense_keys))
    lines.append(
        f"# dataset={dataset_name} train={len(Y_train)} test={len(Y_test)} epochs={ep} seed={SEED} quick={quick} legacy_defenses={dk_note}"
    )
    lines.append(f"# swap_note: {note_sw}")
    lines.append(f"# clean_acc_pct: {acc_clean * 100:.4f}")
    lines.append("")
    lines.append(f"{'Defense':<40} {'Acc%':>10} {'detect%':>10} {'extra':>20}")
    lines.append("-" * 82)
    for key, title in DEFENSE_ORDER:
        if key not in legacy:
            continue
        r = legacy[key]
        extra = ""
        lines.append(
            f"{title:<40} {r['acc'] * 100:10.2f} {r['detect_rate']:10.1f} {extra:>20}"
        )

    lines.append(
        f"{'RGAR (full recon + trust)':<40} {acc_rgar_full * 100:10.2f} {det_full:10.1f} {_meta_short(meta_full):>20}"
    )
    lines.append(
        f"{'RGAR (downweight-only)':<40} {acc_rgar_dw * 100:10.2f} {det_dw:10.1f} {_meta_short(meta_dw):>20}"
    )

    body = "\n".join(lines) + "\n"
    print(body, flush=True)
    root = os.path.dirname(os.path.abspath(out_path)) or "."
    if root and not os.path.isdir(root):
        os.makedirs(root, exist_ok=True)
    mode = "a" if os.path.isfile(out_path) else "w"
    with open(out_path, mode, encoding="utf-8") as f:
        f.write(body)


def main():
    p = argparse.ArgumentParser(description="VFL attack experiments")
    p.add_argument("--dataset", type=str, default="MNIST", help="Dataset name")
    p.add_argument(
        "--experiment",
        type=str,
        choices=["suite", "sensitivity", "strategies", "compare_rgar"],
        default="suite",
        help="compare_rgar=legacy DEFENSE_ORDER + RGAR table -> results_rgar_compare.txt",
    )
    p.add_argument(
        "--rgar_out",
        type=str,
        default="results_rgar_compare.txt",
        help="Output path for compare_rgar experiment",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Fewer epochs for smoke tests (compare_rgar only)",
    )
    p.add_argument(
        "--compare_defenses",
        type=str,
        default="all",
        help="compare_rgar: comma-separated defense keys (subset of suite) or 'all'",
    )
    p.add_argument("--train_samples", type=int, default=5000)
    p.add_argument("--test_samples", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--k", type=int, nargs="+", default=[1, 2, 3, 5, 7], help="k values for sensitivity")
    p.add_argument("--use_signature", action="store_true", help="Use cluster signature in cache (swap_strategies style)")
    args = p.parse_args()

    # One-line visibility: training uses attack_core.DEVICE (cuda if torch was built with CUDA).
    tv_cuda = getattr(torch.version, "cuda", None)
    print(
        f"[VFL] PyTorch {torch.__version__} | runtime device: {DEVICE}"
        + (f" | torch.version.cuda={tv_cuda}" if tv_cuda else " | torch.version.cuda=None (CPU wheel)"),
        flush=True,
    )
    if DEVICE == "cpu":
        print(
            "[VFL] WARNING: Running on CPU. If you have an NVIDIA GPU, your install is likely CPU-only.",
            "[VFL] Fix: pip uninstall torch torchvision -y && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124",
            sep="\n",
            flush=True,
        )
    else:
        print(f"[VFL] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    print(f"Dataset: {args.dataset}", flush=True)
    XA_train, XB_train, Y_train, XA_test, XB_test, Y_test = load_data(
        args.dataset, args.train_samples, args.test_samples
    )
    print(f"Train: {len(Y_train)}, Test: {len(Y_test)}", flush=True)
    # CIFAR-10 uses SOTA training with more epochs by default
    epochs = CIFAR10_EPOCHS if args.dataset.upper() == "CIFAR10" and args.epochs == EPOCHS else args.epochs
    if epochs != args.epochs:
        print(f"Using CIFAR10 SOTA default: {epochs} epochs", flush=True)

    if args.experiment == "suite":
        run_suite(
            args.dataset,
            XA_train, XB_train, Y_train,
            XA_test, XB_test, Y_test,
            epochs=epochs,
        )
    elif args.experiment == "sensitivity":
        run_sensitivity(
            args.dataset,
            XA_train, XB_train, Y_train,
            XA_test, XB_test, Y_test,
            k_values=args.k,
            epochs=epochs,
        )
    elif args.experiment == "strategies":
        run_strategies(
            args.dataset,
            XA_train, XB_train, Y_train,
            XA_test, XB_test, Y_test,
            epochs=epochs,
            use_signature=args.use_signature,
        )
    else:
        legacy_keys: Optional[Set[str]] = None
        if args.compare_defenses.strip().lower() != "all":
            legacy_keys = {x.strip() for x in args.compare_defenses.split(",") if x.strip()}
            valid = {k for k, _ in DEFENSE_ORDER}
            bad = legacy_keys - valid
            if bad:
                raise SystemExit(f"Unknown --compare_defenses keys: {sorted(bad)}. Valid: {sorted(valid)}")
        run_compare_rgar(
            args.dataset,
            XA_train,
            XB_train,
            Y_train,
            XA_test,
            XB_test,
            Y_test,
            epochs=epochs,
            out_path=args.rgar_out,
            quick=args.quick,
            defense_keys=legacy_keys,
        )


if __name__ == "__main__":
    main()
