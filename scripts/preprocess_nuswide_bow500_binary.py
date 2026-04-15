#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np


def _read_label_file(path: str) -> np.ndarray:
    y = np.loadtxt(path, dtype=np.int64)
    return y.reshape(-1)


def _load_bow_dat(path: str, dim: int = 500) -> np.ndarray:
    """
    NUS-WIDE BoW_int files are distributed as .dat. Different mirrors package them
    as whitespace text or raw binary. We try text first, then int32/float32 binary.
    """
    # Try text
    try:
        X = np.loadtxt(path, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != dim:
            raise ValueError(f"Text parse produced shape {X.shape}, expected second dim={dim}")
        return X
    except Exception:
        pass

    # Try binary int32 then float32
    data = np.fromfile(path, dtype=np.int32)
    if data.size % dim == 0 and data.size > 0:
        X = data.reshape(-1, dim).astype(np.float32)
        return X

    data = np.fromfile(path, dtype=np.float32)
    if data.size % dim == 0 and data.size > 0:
        X = data.reshape(-1, dim).astype(np.float32)
        return X

    raise RuntimeError(f"Could not parse BoW file at {path}. Expected text or binary with dim={dim}.")


def _select_binary(
    X: np.ndarray,
    y_pos: np.ndarray,
    y_neg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    y_pos = (y_pos > 0).astype(np.int64)
    y_neg = (y_neg > 0).astype(np.int64)
    keep = (y_pos + y_neg) == 1  # XOR: exactly one concept
    Xs = X[keep]
    ys = y_pos[keep]  # 1 if pos concept, else 0
    return Xs, ys


def main() -> int:
    p = argparse.ArgumentParser(description="Preprocess NUS-WIDE BoW_int into binary classification NPZ")
    p.add_argument("--nuswide_root", type=str, required=True, help="Path to extracted NUS-WIDE root folder")
    p.add_argument("--concept_pos", type=str, default="sky")
    p.add_argument("--concept_neg", type=str, default="clouds")
    p.add_argument("--out_npz", type=str, default="./data/nuswide/nuswide_bow500_binary.npz")
    args = p.parse_args()

    root = args.nuswide_root

    # You should drop/extract these raw folders:
    # - Groundtruth/TrainTestLabels/Labels_<concept>_{Train,Test}.txt
    # - Low_Level_Features/BoW_int/BoW_{Train,Test}_int.dat (folder name may vary)
    gt = os.path.join(root, "Groundtruth", "TrainTestLabels")
    y_pos_tr = _read_label_file(os.path.join(gt, f"Labels_{args.concept_pos}_Train.txt"))
    y_pos_te = _read_label_file(os.path.join(gt, f"Labels_{args.concept_pos}_Test.txt"))
    y_neg_tr = _read_label_file(os.path.join(gt, f"Labels_{args.concept_neg}_Train.txt"))
    y_neg_te = _read_label_file(os.path.join(gt, f"Labels_{args.concept_neg}_Test.txt"))

    # Locate BoW files (support common official naming)
    candidates = [
        ("BoW_Train_int.dat", "BoW_Test_int.dat"),
        ("BoW_Train_int.dat", "BoW_Test_int.dat"),
    ]
    bow_dir_candidates = [
        os.path.join(root, "Low_Level_Features"),
        os.path.join(root, "LowLevelFeatures"),
        os.path.join(root, "Low_Level_Features", "BoW_int"),
        os.path.join(root, "Low_Level_Features", "BoW"),
        os.path.join(root, "NUS_WIDE_Low_Level_Features"),
        os.path.join(root, "NUS_WIDE_Low_Level_Features", "BoW_int"),
        os.path.join(root, "NUS_WIDE_Low_Level_Features", "BoW"),
    ]

    bow_train = bow_test = None
    for d in bow_dir_candidates:
        for tr_name, te_name in candidates:
            trp = os.path.join(d, tr_name)
            tep = os.path.join(d, te_name)
            if os.path.isfile(trp) and os.path.isfile(tep):
                bow_train, bow_test = trp, tep
                break
        if bow_train:
            break
    if bow_train is None:
        raise SystemExit(
            "Could not find BoW_Train_int.dat / BoW_Test_int.dat under nuswide_root. "
            "Place the official Low-Level Features there."
        )

    Xtr = _load_bow_dat(bow_train, dim=500)
    Xte = _load_bow_dat(bow_test, dim=500)

    if len(Xtr) != len(y_pos_tr) or len(Xte) != len(y_pos_te):
        raise SystemExit(
            f"Row mismatch: Xtr={len(Xtr)} ytr={len(y_pos_tr)} | Xte={len(Xte)} yte={len(y_pos_te)}. "
            "Make sure BoW and Groundtruth Train/Test splits match."
        )

    Xtr_s, ytr_s = _select_binary(Xtr, y_pos_tr, y_neg_tr)
    Xte_s, yte_s = _select_binary(Xte, y_pos_te, y_neg_te)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_npz)), exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        X_train=Xtr_s.astype(np.float32),
        y_train=ytr_s.astype(np.int64),
        X_test=Xte_s.astype(np.float32),
        y_test=yte_s.astype(np.int64),
        concept_pos=args.concept_pos,
        concept_neg=args.concept_neg,
        feature="BoW_int_500",
        bow_train_path=bow_train,
        bow_test_path=bow_test,
    )
    print(f"[OK] Wrote {args.out_npz} | train={len(ytr_s)} test={len(yte_s)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

