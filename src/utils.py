import numpy as np
import multiprocessing as mp
import scipy.sparse as sp


def get_params_for_mp(n_triples):
    n_cores = mp.cpu_count()
    pool = mp.Pool(n_cores)
    avg = n_triples // n_cores

    range_list = []
    start = 0
    for i in range(n_cores):
        num = avg + 1 if i < n_triples - avg * n_cores else avg
        range_list.append([start, start + num])
        start += num

    return n_cores, pool, range_list


def one_hot_path_id(train_paths, valid_paths, test_paths):
    path_dict = {}
    n_paths = 0

    res = []
    for data in (train_paths, valid_paths, test_paths):
        # bag of paths
        bop_list = []
        for paths in data:
            bop = []
            for path in paths:
                path = tuple(path)
                if path not in path_dict:
                    path_dict[path] = n_paths
                    n_paths += 1
                bop.append(path_dict[path])
            bop_list.append(bop)
        res.append(bop_list)

    return [get_sparse_feature_matrix(bop_list, n_paths) for bop_list in res], n_paths


def get_sparse_feature_matrix(non_zeros, n_cols):
    features = sp.lil_matrix((len(non_zeros), n_cols), dtype=np.float64)
    for i in range(len(non_zeros)):
        for j in non_zeros[i]:
            features[i, j] = +1.0
    return features


def sparse_to_tuple(sparse_matrix):
    if not sp.isspmatrix_coo(sparse_matrix):
        sparse_matrix = sparse_matrix.tocoo()
    indices = np.vstack((sparse_matrix.row, sparse_matrix.col)).transpose()
    values = sparse_matrix.data
    shape = sparse_matrix.shape
    return indices, values, shape
