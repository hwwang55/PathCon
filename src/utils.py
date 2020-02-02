import numpy as np
import multiprocessing as mp
import scipy.sparse as sp


def count_paths_with_mp(train_set, e2re, max_path_len, data):
    n_cores, pool, range_list = get_params_for_mp(len(data))
    results = pool.map(count_paths, zip([train_set] * n_cores,
                                        [e2re] * n_cores,
                                        [max_path_len] * n_cores,
                                        [data[i[0]:i[1]] for i in range_list],
                                        range(n_cores)))

    # sort the results by pid to make sure that data preserve the original order
    sorted_results = sorted(results, key=lambda x: x[1])

    paths_list = []
    for paths_sublist, _ in sorted_results:
        paths_list.extend(paths_sublist)

    return paths_list


def get_params_for_mp(n_triples):
    n_cores = max(mp.cpu_count(), 8)
    pool = mp.Pool(n_cores)
    avg = n_triples // n_cores

    range_list = []
    start = 0
    for i in range(n_cores):
        num = avg + 1 if i < n_triples - avg * n_cores else avg
        range_list.append([start, start + num])
        start += num

    return n_cores, pool, range_list


def count_paths(inputs):
    train_set, e2re, max_path_len, data, pid = inputs
    paths_list = []
    for head, tail, relation in data:
        paths = bfs(head, tail, relation, e2re, max_path_len, (tail, head, relation) in train_set)
        paths_list.append(paths)
    return paths_list, pid


def bfs(head, tail, relation, e2re, max_path_len, flag):
    # put length-1 paths into all_paths except (relation, tail) if exists
    # each element in all_paths is a path consisting of a sequence of (relation, entity)
    all_paths = [[i] for i in e2re[head] - {(relation, tail)}]

    p = 0
    for length in range(2, max_path_len + 1):
        while p < len(all_paths) and len(all_paths[p]) < length:
            path = all_paths[p]
            last_entity_in_path = path[-1][1]
            if last_entity_in_path != tail:
                entities_in_path = set([head] + [i[1] for i in path])
                for edge in e2re[last_entity_in_path]:
                    # append (relation, entity) to the path if the new entity does not appear in this path before
                    if edge[1] not in entities_in_path:
                        all_paths.append(path + [edge])
            p += 1

    paths = []
    path_set = set()
    # TODO
    remove_duplicate_paths = True

    for path in all_paths:
        # if this path ends at tail
        if path[-1][1] == tail:
            candidate_path = [i[0] for i in path]
            if remove_duplicate_paths:
                if tuple(candidate_path) not in path_set:
                    paths.append(candidate_path)
                    path_set.add(tuple(candidate_path))
            else:
                paths.append(candidate_path)

    # if the reverse edge is in the KG, add it back
    if flag:
        paths.append([relation])

    return paths


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


def sample_paths(train_paths, valid_paths, test_paths, null_relation, max_path_len, path_samples):
    path2id = {}
    id2path = []
    id2length = []
    n_paths = 0

    res = []
    for data in [train_paths, valid_paths, test_paths]:
        path_ids_for_data = []
        for paths in data:
            path_ids_for_triplet = []

            for path in paths:
                path_tuple = tuple(path)
                if path_tuple not in path2id:
                    path2id[path_tuple] = n_paths
                    id2length.append(len(path))
                    id2path.append(path + [null_relation] * (max_path_len - len(path)))  # padding
                    n_paths += 1
                path_ids_for_triplet.append(path2id[path_tuple])

            sampled_path_ids_for_triplet = np.random.choice(path_ids_for_triplet,
                                                            size=path_samples,
                                                            replace=len(path_ids_for_triplet) < path_samples)
            path_ids_for_data.append(sampled_path_ids_for_triplet)

        path_ids_for_data = np.array(path_ids_for_data, dtype=np.int32)
        res.append(path_ids_for_data)

    return res, id2path, id2length


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
