import re
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer


args = None

entity_dict = {}
relation_dict = {}

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
    file = open(file_name)
    for line in file:
        index, name = line.strip().split('\t')
        entity_dict[name] = int(index)
    file.close()


def process_relations(file_name):
    bow = []
    count_vec = CountVectorizer()

    file = open(file_name)
    for line in file:
        index, name = line.strip().split('\t')
        relation_dict[name] = int(index)

        if args.feature == 'bow':
            tokens = re.findall('[a-z]{2,}', name)
            bow.append(' '.join(tokens))
    file.close()

    if args.feature == 'bow':
        bow = count_vec.fit_transform(bow)
        np.save('../data/' + args.dataset + '/bow.npy', bow.toarray())


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


def process_kg(train_data, n_entity, n_relation):
    for edge_idx, triplet in enumerate(train_data):
        head_idx, tail_idx, relation_idx = triplet

        if args.use_ls:
            entity2edge_set[head_idx].add(edge_idx)
            entity2edge_set[tail_idx].add(edge_idx)
            edge2entities.append([head_idx, tail_idx])
            edge2relation.append(relation_idx)

        if args.use_e2e:
            e2re[head_idx].add((relation_idx, tail_idx))
            e2re[tail_idx].add((relation_idx, head_idx))

    # To handle the case where a node does not appear in the training data (i.e., this node has no neighboring edge),
    # we introduce a null entity (ID: n_entities), a null edge (ID: n_edges), and a null relation (ID: n_relations).
    # entity2edge_set[isolated_node] = {null_edge}
    # entity2edge_set[null_entity] = {null_edge}
    # edge2entities[null_edge] = [null_entity, null_entity]
    # edge2relation[null_edge] = null_relation
    # The feature of null_relation is a zero vector. See _build_model() of model.py for details

    if args.use_ls:
        null_entity = n_entity
        null_relation = n_relation
        null_edge = len(edge2entities)
        edge2entities.append([null_entity, null_entity])
        edge2relation.append(null_relation)

        for i in range(n_entity + 1):
            if i not in entity2edge_set:
                entity2edge_set[i] = {null_edge}

            sampled_neighbors = np.random.choice(list(entity2edge_set[i]),
                                                 size=args.sample,
                                                 replace=len(entity2edge_set[i]) < args.sample)
            entity2edges.append(sampled_neighbors)


def get_neighbors(data, is_train):
    data_np = np.array(data)
    edge2entities_np = np.array(edge2entities)
    edge2relation_np = np.array(edge2relation)
    entity2edges_np = np.array(entity2edges)

    # the first element in edge_list is relations of all triples
    edges_list = [data_np[:, 2]]
    null_edge = len(edge2entities_np) - 1

    if is_train:
        for i in range(args.iteration):
            edges_list.append([])

        for edge, (head, tail, _) in enumerate(data_np):
            # save the two rows temporarily
            row_head = entity2edges_np[head]
            row_tail = entity2edges_np[tail]

            # replace the two rows (head and tail) with new sampled neighbors
            entity2edges_np[head] = get_sampled_neighbors(entity2edge_set[head], edge, null_edge, args.sample)
            entity2edges_np[tail] = get_sampled_neighbors(entity2edge_set[tail], edge, null_edge, args.sample)

            for j in range(args.iteration):
                if j == 0:
                    neighbor_entities = np.array([head, tail])
                else:
                    neighbor_entities = edge2entities_np[edges_list[j][edge]].flatten()
                neighbor_edges = entity2edges_np[neighbor_entities].flatten()
                edges_list[j + 1].append(neighbor_edges)

            entity2edges_np[head] = row_head
            entity2edges_np[tail] = row_tail

        for i in range(args.iteration):
            edges_list[i + 1] = np.array(edges_list[i + 1])

    else:
        for i in range(args.iteration):
            if i == 0:
                neighbor_entities = data_np[:, 0:2]
            else:
                neighbor_entities = edge2entities_np[edges_list[-1]].reshape([len(data_np), -1])
            neighbor_edges = entity2edges_np[neighbor_entities].reshape([len(data_np), -1])
            edges_list.append(neighbor_edges)

    # map each edge to its relation type
    for i in range(args.iteration):
        edges_list[i + 1] = edge2relation_np[edges_list[i + 1]]

    return edges_list


def get_sampled_neighbors(neighbor_set, masking_edge, null_edge, sampling_size):
    assert masking_edge in neighbor_set
    neighbor_set_new = neighbor_set - {masking_edge}
    if len(neighbor_set_new) == 0:
        neighbor_set_new = {null_edge}
    sampled_neighbors = np.random.choice(list(neighbor_set_new),
                                         size=sampling_size,
                                         replace=len(neighbor_set_new) < sampling_size)
    return sampled_neighbors


def count_paths(train_set, data, max_path_len):
    paths = []
    for head, tail, relation in data:
        path = bfs(head, tail, relation, max_path_len, train_set)
        paths.append(path)
    return paths


def bfs(head, tail, relation, max_path_len, train_set):
    # put length-1 paths into all_paths except (relation, tail) if exists
    # each element in all_paths is a path consisting of a sequence of (relation, entity)
    all_paths = [[i] for i in e2re[head] - {(relation, tail)}]
    p = 0
    for length in range(2, max_path_len + 1):
        while p < len(all_paths) and len(all_paths[p]) < length:
            path = all_paths[p]
            last_entity_in_path = path[-1][1]
            entities_in_path = [head] + [i[1] for i in path]
            for edge in e2re[last_entity_in_path]:
                # append (relation, entity) to the path if the new entity does not appear in this path before
                if edge[1] not in entities_in_path:
                    all_paths.append(path + [edge])
            p += 1

    res = []
    for path in all_paths:
        # if this path ends at tail
        if path[-1][1] == tail:
            res.append([i[0] for i in path])

    # if the reverse edge is in the KG, add it back
    if (tail, head, relation) in train_set:
        res.append([relation])

    return res


def load_data(model_args):
    global args
    args = model_args
    directory = '../data/' + args.dataset + '/'

    print('reading entity dict and relation dict ...')
    process_entities(directory + 'entities.dict')
    process_relations(directory + 'relations.dict')

    print('reading train, validation, and test data ...')
    train_data = process_data(directory + 'train.txt')
    valid_data = process_data(directory + 'valid.txt')
    test_data = process_data(directory + 'test.txt')

    '''
    remove_symmetric_edges = False
    if remove_symmetric_edges:
        train_set = set()
        for head, tail, relation in train_data:
            if (head, tail, relation) not in train_set and (tail, head, relation) not in train_set:
                train_set.add((head, tail, relation))
        train_data = np.array(list(train_set))
    '''

    print('processing the knowledge graph ...')
    process_kg(train_data, len(entity_dict), len(relation_dict))

    neighbors = []
    paths = []

    if args.use_ls:
        print('sampling neighbor edges ...')
        train_neighbors = get_neighbors(train_data, True)
        valid_neighbors = get_neighbors(valid_data, False)
        test_neighbors = get_neighbors(test_data, False)
        neighbors.extend([train_neighbors, valid_neighbors, test_neighbors])
    else:
        neighbors.extend([None] * 3)

    if args.use_e2e:
        print('counting paths from head to tail ...')
        train_set = set(train_data)
        train_paths = count_paths(train_set, train_data, args.max_path_len)
        valid_paths = count_paths(train_set, valid_data, args.max_path_len)
        test_paths = count_paths(train_set, test_data, args.max_path_len)
        paths.extend([train_paths, valid_paths, test_paths])
    else:
        paths.extend([None] * 3)

    return neighbors, paths, len(relation_dict)
