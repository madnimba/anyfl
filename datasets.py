# datasets.py
# Utility functions for loading and preprocessing datasets

import torch
from torchvision import datasets, transforms
import numpy as np

def load_mnist(n_samples=None):
    tf = transforms.ToTensor()
    mnist = datasets.MNIST('.', train=True, download=True, transform=tf)
    X, Y = [], []
    for i in range(len(mnist) if n_samples is None else n_samples):
        img, lbl = mnist[i]
        X.append(img.numpy())
        Y.append(lbl)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def load_cifar10(n_samples=None):
    tf = transforms.ToTensor()
    cifar = datasets.CIFAR10('.', train=True, download=True, transform=tf)
    X, Y = [], []
    for i in range(len(cifar) if n_samples is None else n_samples):
        img, lbl = cifar[i]
        X.append(img.numpy())
        Y.append(lbl)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y



def load_mushroom(n_samples=None, one_hot=True):
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import OneHotEncoder

    df = fetch_openml('mushroom', version=1, as_frame=True).frame
    # target column is 'class' with values {'e','p'} -> map to {0,1}
    y = (df['class'].astype(str) == 'p').astype(np.int64).to_numpy()
    X_cat = df.drop(columns=['class'])

    if one_hot:
        # scikit-learn >= 1.2 uses sparse_output; older uses sparse
        try:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse=True)
        X = ohe.fit_transform(X_cat)
        if n_samples is not None:
            X = X[:n_samples]
            y = y[:n_samples]
        X = X.toarray().astype(np.float32)
    else:
        # label-encode each categorical column (not recommended for distance-based models)
        X = X_cat.apply(lambda col: col.astype('category').cat.codes).to_numpy(dtype=np.int64)
        if n_samples is not None:
            X = X[:n_samples]
            y = y[:n_samples]
        X = X.astype(np.float32)

    return X, y


def load_bank(variant="full", n_samples=None, drop_duration=False):
    """
    UCI Bank Marketing from OpenML.
    variant: "full" (bank-marketing, ~45k) or "additional" (bank-additional, ~41k).
    Returns: X (float32), y (int64)
    """
    import pandas as pd
    from sklearn.datasets import fetch_openml

    # Try a few OpenML identifiers to be robust
    tries = []
    if variant.lower() == "full":
        tries += [dict(name="bank-marketing", version=1), dict(data_id=1461),
                  dict(name="bank-marketing", version=2), dict(data_id=1558)]
    else:  # "additional"
        tries += [dict(name="bank-additional", version=1), dict(data_id=1509),
                  dict(name="bank-additional", version=2)]

    last_err = None
    for kw in tries:
        try:
            X_df, y_ser = fetch_openml(as_frame=True, return_X_y=True, **kw)
            # labels: map yes/1/true/t to 1, else 0
            y = y_ser.astype(str).str.lower().isin(["yes", "1", "true", "t"]).astype(np.int64).to_numpy()
            break
        except Exception as e:
            last_err = e
            X_df, y = None, None
    if X_df is None:
        raise RuntimeError(f"Could not fetch Bank Marketing ({variant}). Last error: {last_err}")

    # Optional leakage guard (match your clustering script default behavior)
    if drop_duration and "duration" in X_df.columns:
        X_df = X_df.drop(columns=["duration"])

    # Flip polarity if positives dominate (keeps minority as 1 like your script)
    if y.mean() > 0.5:
        y = 1 - y

    # Coerce numeric-looking object columns to numeric
    obj_cols = X_df.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        coerced = pd.to_numeric(X_df[c], errors="coerce")
        if coerced.notna().mean() >= 0.95:
            X_df[c] = coerced

    # One-hot only the categorical cols, keep numeric as-is (dense)
    num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_df.columns if c not in num_cols]

    if len(cat_cols):
        X_cat = pd.get_dummies(X_df[cat_cols], drop_first=False)
        X_num = X_df[num_cols].reset_index(drop=True)
        X = pd.concat([X_num, X_cat], axis=1).to_numpy(dtype=np.float32)
    else:
        X = X_df.to_numpy(dtype=np.float32)

    if n_samples is not None:
        X = X[:n_samples]
        y = y[:n_samples]

    return X, y


def load_ucihar(n_samples=None):
    """
    UCI HAR (Human Activity Recognition) via OpenML.
    Returns:
        X : np.ndarray, shape [N, D], float32
        y : np.ndarray, shape [N], int64 (labels 0..C-1)
    """
    import pandas as pd
    from sklearn.datasets import fetch_openml

    tries = [
        dict(name="har", version=1),                 # common alias on OpenML
        dict(data_id=1478),                          # known dataset id
        dict(name="Human Activity Recognition Using Smartphones"),
        dict(name="UCI HAR Dataset"),
    ]
    last_err = None
    for kw in tries:
        try:
            X_df, y_ser = fetch_openml(as_frame=True, return_X_y=True, **kw)

            # Ensure numeric features (HAR should already be numeric, but guard anyway)
            for c in X_df.columns:
                if not np.issubdtype(X_df[c].dtype, np.number):
                    X_df[c] = pd.to_numeric(X_df[c], errors="coerce").fillna(0.0)

            X = X_df.to_numpy(dtype=np.float32)

            # Factorize labels to 0..C-1 (OpenML may provide strings like 'WALKING', etc.)
            y_codes, _ = pd.factorize(y_ser.astype(str), sort=True)
            y = y_codes.astype(np.int64)

            if n_samples is not None:
                X = X[:n_samples]
                y = y[:n_samples]
            return X, y
        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Could not fetch UCI HAR from OpenML. Last error: {last_err}")


def get_dataset(name, n_samples=None):
    if name.lower() == 'mnist':
        return load_mnist(n_samples)
    elif name.lower() == 'cifar10':
        return load_cifar10(n_samples)
    elif name.lower() == '20ng':
        # Placeholder for 20 Newsgroups dataset loading
        from sklearn.datasets import fetch_20newsgroups
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        newsgroups = fetch_20newsgroups(subset='all')
        vectorizer = TfidfVectorizer(max_features=2000)
        X = vectorizer.fit_transform(newsgroups.data).toarray()
        Y = newsgroups.target
        if n_samples is not None:
            X = X[:n_samples]
            Y = Y[:n_samples]
        return X, Y
    
    elif name.lower() == 'mushroom':
        return load_mushroom(n_samples, one_hot=True)
    elif name.lower() == 'bank':
        return load_bank(variant="full", n_samples=n_samples, drop_duration=True)
    elif name.lower() == 'bank-additional':
        return load_bank(variant="additional", n_samples=n_samples, drop_duration=True)
    elif name.lower() in ('uci_har', 'har', 'human activity recognition'):
        return load_ucihar(n_samples)
    else:
        raise ValueError(f"Unknown dataset: {name}")
