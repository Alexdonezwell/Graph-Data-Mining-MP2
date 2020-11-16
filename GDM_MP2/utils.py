import sys
import pickle as pkl
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import glob


def label_encoding(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def normalize(mx):
    '''Row-normalize sparse matrix'''
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    '''Convert a scipy sparse matrix to a torch sparse tensor.'''
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_data(dataset, seed=None):
    if seed:
        torch.manual_seed(seed)
        rg = np.random.default_rng(seed)
    raw_X_Y = np.genfromtxt('data/{}/{}.content'.format(dataset, dataset), dtype=np.dtype(str))
    idx = np.array(raw_X_Y[:, 0], dtype=np.dtype(str))
    X = sp.csr_matrix(raw_X_Y[:, 1:-1], dtype=np.float32)
    Y = label_encoding(raw_X_Y[:, -1])

    # build graph
    idx_map = {j: i for i, j in enumerate(idx)}
    raw_A = np.genfromtxt('data/{}/{}.cites'.format(dataset, dataset), dtype=np.dtype(str))
    
    diff = np.setdiff1d(raw_A.flatten(), idx)
    A_pairs = raw_A[~np.isin(raw_A, diff).any(axis=1)]
    A_pairs = np.vectorize(idx_map.get)(A_pairs).astype(np.int32)
    A = sp.coo_matrix((np.ones(A_pairs.shape[0]), (A_pairs[:, 0], A_pairs[:, 1])),
                        shape=(Y.shape[0], Y.shape[0]),
                        dtype=np.float32)

    # build symmetric Adjacency matrix
    A = A + A.T.multiply(A.T > A) - A.multiply(A.T > A)

    X = normalize(X)
    A = normalize(A + sp.eye(A.shape[0]))

    sample_num = X.shape[0]

    idx_all = list(range(sample_num))
    rg.shuffle(idx_all)
    idx_train = idx_all[: int(sample_num * 0.7)]
    idx_val = idx_all[int(sample_num * 0.7): int(sample_num * 0.9)]
    idx_test = idx_all[int(sample_num * 0.9): ]

    X = torch.FloatTensor(np.array(X.todense()))
    Y = torch.LongTensor(np.where(Y)[1])
    A = sparse_mx_to_torch_sparse_tensor(A)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return A, X, Y, idx_train, idx_val, idx_test


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

