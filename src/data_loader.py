import re
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from utils import get_params_for_mp, one_hot_path_id


# entity index -> set of (both incoming and outgoing) edges connecting to this entity
entity2edge_set = defaultdict(set)

# each row in entity2edges is the sampled edges connecting to this entity
entity2edges = []

# each row in edge2entities is the two entities connected by this edge
edge2entities = []

# each row in edge2relation is the relation type of this edge
edge2relation = []

# entity index -> set of pair (relation, entity) connecting to this entity
e2re = defaultdict(set)


def process_entities(file_name):
    d = {}
    file = open(file_name)
    for line in file:
        index, name = line.strip().split('\t')
        d[name] = int(index)
    file.close()

    return d


def process_relations(file_name):
    bow = []
    count_vec = CountVectorizer()

    d = {}
    file = open(file_name)
    for line in file:
        index, name = line.strip().split('\t')
        d[name] = int(index)

        if args.feature_mode == 'bow':
            tokens = re.findall('[a-z]{2,}', name)
            bow.append(' '.join(tokens))
    file.close()

    if args.feature_mode == 'bow':
        bow = count_vec.fit_transform(bow)
        np.save('../data/' + args.dataset + '/bow.npy', bow.toarray())

    return d


def process_data(file_name):
    data = []

    file = open(file_name)
    for line in file:
        head, relation, tail = line.strip().split('\t')

        head_idx = entity_dict[head]
        relation_idx = relation_dict[relation]
        tail_idx = entity_dict[tail]

        data.append((head_idx, tail_idx, relation_idx))
    file.close()

    return data


def process_kg(train_data):
    for edge_idx, triplet in enumerate(train_data):
        head_idx, tail_idx, relation_idx = triplet

        if args.use_gnn:
            entity2edge_set[head_idx].add(edge_idx)
            entity2edge_set[tail_idx].add(edge_idx)
            edge2entities.append([head_idx, tail_idx])
            edge2relation.append(relation_idx)

        if args.use_path:
            e2re[head_idx].add((relation_idx, tail_idx))
            e2re[tail_idx].add((relation_idx, head_idx))

    # To handle the case where a node does not appear in the training data (i.e., this node has no neighboring edge),
    # we introduce a null entity (ID: n_entities), a null edge (ID: n_edges), and a null relation (ID: n_relations).
    # entity2edge_set[isolated_node] = {null_edge}
    # entity2edge_set[null_entity] = {null_edge}
    # edge2entities[null_edge] = [null_entity, null_entity]
    # edge2relation[null_edge] = null_relation
    # The feature of null_relation is a zero vector. See _build_model() of model.py for details

    if args.use_gnn:
        null_entity = len(entity_dict)
        null_relation = len(relation_dict)
        null_edge = len(edge2entities)
        edge2entities.append([null_entity, null_entity])
        edge2relation.append(null_relation)

        for i in range(len(entity_dict) + 1):
            if i not in entity2edge_set:
                entity2edge_set[i] = {null_edge}

            sampled_neighbors = np.random.choice(list(entity2edge_set[i]),
                                                 size=args.neighbor_samples,
                                                 replace=len(entity2edge_set[i]) < args.neighbor_samples)
            entity2edges.append(sampled_neighbors)


def get_neighbors_for_train(inputs):
    data, edge_offset, pid = inputs

    data_np = np.array(data)
    edge2entities_np = np.array(edge2entities)
    edge2relation_np = np.array(edge2relation)
    entity2edges_np = np.array(entity2edges)

    # the first element in edge_list is relations of all triples
    edges_list = [data_np[:, 2]]

    for i in range(args.gnn_layers):
        edges_list.append([])

    for i, (head, tail, _) in enumerate(data_np):
        edge = i + edge_offset
        # store the two rows temporarily
        row_head = entity2edges_np[head]
        row_tail = entity2edges_np[tail]

        # replace the two rows (head and tail) with new sampled neighbors
        entity2edges_np[head] = sample_neighbors(entity2edge_set[head], edge, args.neighbor_samples)
        entity2edges_np[tail] = sample_neighbors(entity2edge_set[tail], edge, args.neighbor_samples)

        for j in range(args.gnn_layers):
            if j == 0:
                neighbor_entities = np.array([head, tail])
            else:
                neighbor_entities = edge2entities_np[edges_list[j][i]].flatten()
            neighbor_edges = entity2edges_np[neighbor_entities].flatten()
            edges_list[j + 1].append(neighbor_edges)

        # restore the two rows
        entity2edges_np[head] = row_head
        entity2edges_np[tail] = row_tail

    for i in range(args.gnn_layers):
        edges_list[i + 1] = np.array(edges_list[i + 1])

    # map each edge to its relation type
    for i in range(args.gnn_layers):
        edges_list[i + 1] = edge2relation_np[edges_list[i + 1]]

    return edges_list, pid


def get_neighbors_for_train_with_mp(data):
    n_cores, pool, range_list = get_params_for_mp(len(data))
    results = pool.map(get_neighbors_for_train, zip([data[i[0]:i[1]] for i in range_list],
                                                    [i[0] for i in range_list],
                                                    range(n_cores)))

    # sort the results by pid to make sure that train data preserve the original order
    sorted_results = sorted(results, key=lambda x: x[1])
    edges_list = [np.concatenate([j[i] for j, _ in sorted_results], axis=0) for i in range(args.gnn_layers + 1)]

    return edges_list


def get_neighbors_for_eval(data):
    data_np = np.array(data)
    edge2entities_np = np.array(edge2entities)
    edge2relation_np = np.array(edge2relation)
    entity2edges_np = np.array(entity2edges)

    # the first element in edge_list is relations of all triples
    edges_list = [data_np[:, 2]]

    for i in range(args.gnn_layers):
        if i == 0:
            neighbor_entities = data_np[:, 0:2]
        else:
            neighbor_entities = edge2entities_np[edges_list[-1]].reshape([len(data_np), -1])
        neighbor_edges = entity2edges_np[neighbor_entities].reshape([len(data_np), -1])
        edges_list.append(neighbor_edges)

    # map each edge to its relation type
    for i in range(args.gnn_layers):
        edges_list[i + 1] = edge2relation_np[edges_list[i + 1]]

    return edges_list


def sample_neighbors(neighbor_set, masking_edge, sampling_size):
    null_edge = len(edge2entities) - 1

    assert masking_edge in neighbor_set
    neighbor_set_new = neighbor_set - {masking_edge}
    if len(neighbor_set_new) == 0:
        neighbor_set_new = {null_edge}
    sampled_neighbors = np.random.choice(list(neighbor_set_new),
                                         size=sampling_size,
                                         replace=len(neighbor_set_new) < sampling_size)
    return sampled_neighbors


def count_paths(inputs):
    train_set, data, pid = inputs
    paths_list = []
    for head, tail, relation in data:
        paths = bfs(head, tail, relation, (tail, head, relation) in train_set)
        paths_list.append(paths)
    return paths_list, pid


def count_paths_with_mp(train_set, data):
    n_cores, pool, range_list = get_params_for_mp(len(data))
    results = pool.map(count_paths, zip([train_set] * n_cores, [data[i[0]:i[1]] for i in range_list], range(n_cores)))

    # sort the results by pid to make sure that data preserve the original order
    sorted_results = sorted(results, key=lambda x: x[1])

    paths_list = []
    for paths_sublist, _ in sorted_results:
        paths_list.extend(paths_sublist)

    return paths_list


def bfs(head, tail, relation, flag):
    # put length-1 paths into all_paths except (relation, tail) if exists
    # each element in all_paths is a path consisting of a sequence of (relation, entity)
    all_paths = [[i] for i in e2re[head] - {(relation, tail)}]

    p = 0
    for length in range(2, args.max_path_len + 1):
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

    for path in all_paths:
        # if this path ends at tail
        if path[-1][1] == tail:
            paths.append([i[0] for i in path])

    # if the reverse edge is in the KG, add it back
    if flag:
        paths.append([relation])

    # if using rnn and no path is found for the triplet, put a empty path into paths
    if args.path_mode == 'rnn' and len(paths) == 0:
        paths.append([])

    return paths


def sample_paths(train_data, valid_data, test_data):
    path2id = {}
    id2path = []
    id2length = []
    n_paths = 0

    res = []
    for data in [train_data, valid_data, test_data]:
        path_ids_for_data = []
        for paths in data:
            # consider path frequency by setting the following variable as a list, otherwise setting this as a set
            path_ids_for_triplet = set()

            for path in paths:
                path_tuple = tuple(path)
                if path_tuple not in path2id:
                    path2id[path_tuple] = n_paths
                    id2length.append(len(path))
                    id2path.append(path + [len(relation_dict)] * (args.max_path_len - len(path)))  # padding
                    n_paths += 1
                path_ids_for_triplet.add(path2id[path_tuple])

            sampled_path_ids_for_triplet = np.random.choice(list(path_ids_for_triplet),
                                                            size=args.path_samples,
                                                            replace=len(path_ids_for_triplet) < args.path_samples)
            path_ids_for_data.append(sampled_path_ids_for_triplet)

        path_ids_for_data = np.array(path_ids_for_data)
        res.append(path_ids_for_data)

    return res, n_paths, id2path, id2length


def load_data(model_args):
    global args, entity_dict, relation_dict
    args = model_args
    directory = '../data/' + args.dataset + '/'

    print('reading entity dict and relation dict ...')
    entity_dict = process_entities(directory + 'entities.dict')
    relation_dict = process_relations(directory + 'relations.dict')

    print('reading train, validation, and test data ...')
    train_data = process_data(directory + 'train.txt')
    valid_data = process_data(directory + 'valid.txt')
    test_data = process_data(directory + 'test.txt')

    print('processing the knowledge graph ...')
    process_kg(train_data)

    if args.use_gnn:
        print('sampling neighbor edges ...')

        use_mp = False
        if use_mp:
            train_neighbors = get_neighbors_for_train_with_mp(train_data)
        else:
            train_neighbors, _ = get_neighbors_for_train((train_data, 0, 0))

        valid_neighbors = get_neighbors_for_eval(valid_data)
        test_neighbors = get_neighbors_for_eval(test_data)

        neighbors = [train_neighbors, valid_neighbors, test_neighbors]
    else:
        neighbors = [None] * 3

    params_for_paths = []

    if args.use_path:
        print('counting paths from head to tail ...')
        train_set = set(train_data)

        use_mp = False
        if use_mp:
            train_paths = count_paths_with_mp(train_set, train_data)
            valid_paths = count_paths_with_mp(train_set, valid_data)
            test_paths = count_paths_with_mp(train_set, test_data)
        else:
            train_paths, _ = count_paths((train_set, train_data, 0))
            valid_paths, _ = count_paths((train_set, valid_data, 0))
            test_paths, _ = count_paths((train_set, test_data, 0))

        if args.path_mode == 'id':
            print('transforming paths to one hot IDs ...')
            paths, n_paths = one_hot_path_id(train_paths, valid_paths, test_paths)
            params_for_paths.append(n_paths)
        elif args.path_mode == 'rnn':
            print('sampling paths ...')
            paths, n_paths, id2path, id2length = sample_paths(train_paths, valid_paths, test_paths)
            params_for_paths.extend([n_paths, id2path, id2length])
        else:
            raise ValueError('unknown path embedding mode')
    else:
        paths = [None] * 3

    labels = [np.array([triplet[2] for triplet in train_data]),
              np.array([triplet[2] for triplet in valid_data]),
              np.array([triplet[2] for triplet in test_data])]

    return neighbors, paths, labels, len(relation_dict), params_for_paths
