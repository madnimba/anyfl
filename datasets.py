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
    else:
        raise ValueError(f"Unknown dataset: {name}")
