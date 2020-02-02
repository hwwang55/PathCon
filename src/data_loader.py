import re
import os
import pickle
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from utils import *


# entity index -> set of (both incoming and outgoing) edges connecting to this entity
entity2edge_set = defaultdict(set)

# each row in edge2entities is the two entities connected by this edge
edge2entities = []

# each row in edge2relation is the relation type of this edge
edge2relation = []

# entity index -> set of pair (relation, entity) connecting to this entity
e2re = defaultdict(set)


def read_entities(file_name):
    d = {}
    file = open(file_name)
    for line in file:
        index, name = line.strip().split('\t')
        d[name] = int(index)
    file.close()

    return d


def read_relations(file_name):
    bow = []
    count_vec = CountVectorizer()

    d = {}
    file = open(file_name)
    for line in file:
        index, name = line.strip().split('\t')
        d[name] = int(index)

        if args.feature_mode == 'bow' and not os.path.exists('../data/' + args.dataset + '/bow.npy'):
            tokens = re.findall('[a-z]{2,}', name)
            bow.append(' '.join(tokens))
    file.close()

    if args.feature_mode == 'bow' and not os.path.exists('../data/' + args.dataset + '/bow.npy'):
        bow = count_vec.fit_transform(bow)
        np.save('../data/' + args.dataset + '/bow.npy', bow.toarray())

    return d


def read_triplets(file_name):
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


def build_kg(train_data):
    for edge_idx, triplet in enumerate(train_data):
        head_idx, tail_idx, relation_idx = triplet

        if args.use_neighbor:
            entity2edge_set[head_idx].add(edge_idx)
            entity2edge_set[tail_idx].add(edge_idx)
            edge2entities.append([head_idx, tail_idx])
            edge2relation.append(relation_idx)

        if args.use_path:
            e2re[head_idx].add((relation_idx, tail_idx))
            e2re[tail_idx].add((relation_idx, head_idx))

    # To handle the case where a node does not appear in the training data (i.e., this node has no neighbor edge),
    # we introduce a null entity (ID: n_entities), a null edge (ID: n_edges), and a null relation (ID: n_relations).
    # entity2edge_set[isolated_node] = {null_edge}
    # entity2edge_set[null_entity] = {null_edge}
    # edge2entities[null_edge] = [null_entity, null_entity]
    # edge2relation[null_edge] = null_relation
    # The feature of null_relation is a zero vector. See _build_model() of model.py for details

    if args.use_neighbor:
        null_entity = len(entity_dict)
        null_relation = len(relation_dict)
        null_edge = len(edge2entities)
        edge2entities.append([null_entity, null_entity])
        edge2relation.append(null_relation)

        for i in range(len(entity_dict) + 1):
            if i not in entity2edge_set:
                entity2edge_set[i] = {null_edge}


def get_paths(train_triplets, valid_triplets, test_triplets):
    directory = '../data/' + args.dataset + '/cache/'
    length = str(args.max_path_len)

    if not os.path.exists(directory):
        os.mkdir(directory)

    if os.path.exists(directory + 'train_paths_' + length + '.pkl'):
        print('loading paths from files ...')
        train_paths = pickle.load(open(directory + 'train_paths_' + length + '.pkl', 'rb'))
        valid_paths = pickle.load(open(directory + 'valid_paths_' + length + '.pkl', 'rb'))
        test_paths = pickle.load(open(directory + 'test_paths_' + length + '.pkl', 'rb'))

    else:
        print('counting paths from head to tail ...')
        train_set = set(train_triplets)
        train_paths = count_paths_with_mp(train_set, e2re, args.max_path_len, train_triplets)
        valid_paths = count_paths_with_mp(train_set, e2re, args.max_path_len, valid_triplets)
        test_paths = count_paths_with_mp(train_set, e2re, args.max_path_len, test_triplets)

        print('dumping paths to files ...')
        pickle.dump(train_paths, open(directory + 'train_paths_' + length + '.pkl', 'wb'))
        pickle.dump(valid_paths, open(directory + 'valid_paths_' + length + '.pkl', 'wb'))
        pickle.dump(test_paths, open(directory + 'test_paths_' + length + '.pkl', 'wb'))

    # if using rnn and no path is found for the triplet, put an empty path into paths
    if args.path_mode == 'rnn':
        for paths in train_paths + valid_paths + test_paths:
            if len(paths) == 0:
                paths.append([])

    return train_paths, valid_paths, test_paths


def load_data(model_args):
    global args, entity_dict, relation_dict
    args = model_args
    directory = '../data/' + args.dataset + '/'

    print('reading entity dict and relation dict ...')
    entity_dict = read_entities(directory + 'entities.dict')
    relation_dict = read_relations(directory + 'relations.dict')

    print('reading train, validation, and test data ...')
    train_triplets = read_triplets(directory + 'train.txt')
    valid_triplets = read_triplets(directory + 'valid.txt')
    test_triplets = read_triplets(directory + 'test.txt')

    print('processing the knowledge graph ...')
    build_kg(train_triplets)

    triplets = [train_triplets, valid_triplets, test_triplets]

    if args.use_neighbor:
        train_edges = np.array(range(len(train_triplets)), np.int32)
        neighbor_params = [np.array(edge2entities), np.array(edge2relation)]
    else:
        train_edges = None
        neighbor_params = None

    if args.use_path:
        train_paths, valid_paths, test_paths = get_paths(train_triplets, valid_triplets, test_triplets)
        if args.path_mode == 'id':
            print('transforming paths to one hot IDs ...')
            paths, n_paths = one_hot_path_id(train_paths, valid_paths, test_paths)
            path_params = [n_paths]
        elif args.path_mode == 'rnn':
            print('sampling paths ...')
            paths, id2path, id2length = sample_paths(
                train_paths, valid_paths, test_paths, len(relation_dict), args.max_path_len, args.path_samples)
            path_params = [id2path, id2length]
        else:
            raise ValueError('unknown path embedding mode')
    else:
        paths = [None] * 3
        path_params = None

    n_entities = len(entity_dict)
    n_relations = len(relation_dict)

    return triplets, train_edges, entity2edge_set, paths, n_entities, n_relations, neighbor_params, path_params
