import re
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer


def process_entities(file_name):
    print('reading %s' % file_name)
    d = {}

    file = open(file_name)
    for line in file:
        index, name = line.strip().split('\t')
        d[name] = int(index)
    file.close()

    return d


def process_relations(file_name, args):
    print('reading %s' % file_name)
    d = {}
    bow = []
    count_vec = CountVectorizer()

    file = open(file_name)
    for line in file:
        index, name = line.strip().split('\t')
        d[name] = int(index)

        if args.feature == 'bow':
            tokens = re.findall('[a-z]{2,}', name)
            bow.append(' '.join(tokens))
    file.close()

    if args.feature == 'bow':
        bow = count_vec.fit_transform(bow)
        np.save('../data/' + args.dataset + '/bow.npy', bow.toarray())

    return d


def process_kg(file_name, entity_dict, relation_dict, sampling_size):
    print('processing KG')

    # entity index -> set of (both incoming and outgoing) edges connecting to this entity
    entity2edge_set = defaultdict(set)

    # each row in edge2entities is the two entities connected by this edge
    edge2entities = []

    # each row in edge2relation is the relation type of this edge
    edge2relation = []

    file = open(file_name)
    for edge_idx, line in enumerate(file):
        head, relation, tail = line.strip().split('\t')

        head_idx = entity_dict[head]
        relation_idx = relation_dict[relation]
        tail_idx = entity_dict[tail]

        entity2edge_set[head_idx].add(edge_idx)
        entity2edge_set[tail_idx].add(edge_idx)
        edge2entities.append([head_idx, tail_idx])
        edge2relation.append(relation_idx)
    file.close()

    # each row in entity2edges is the sampled edges connecting to this entity
    entity2edges = []

    # To handle the case where a node does not appear in the training data (i.e., this node has no neighboring edge),
    # we introduce a null entity (ID: n_entities), a null edge (ID: n_edges), and a null relation (ID: n_relations).
    # entity2edge_set[isolated_node] = {null_edge}
    # entity2edge_set[null_entity] = {null_edge}
    # edge2entities[null_edge] = [null_entity, null_entity]
    # edge2relation[null_edge] = null_relation
    # The feature of null_relation is a zero vector. See _build_model() of model.py for details

    null_entity = len(entity_dict)
    null_relation = len(relation_dict)
    null_edge = len(edge2entities)
    edge2entities.append([null_entity, null_entity])
    edge2relation.append(null_relation)

    for i in range(len(entity_dict) + 1):
        if i not in entity2edge_set:
            entity2edge_set[i] = {null_edge}

        sampled_neighbors = np.random.choice(list(entity2edge_set[i]),
                                             size=sampling_size,
                                             replace=len(entity2edge_set[i]) < sampling_size)
        entity2edges.append(sampled_neighbors)

    edge2entities = np.array(edge2entities)
    edge2relation = np.array(edge2relation)
    entity2edges = np.array(entity2edges)

    return edge2entities, edge2relation, entity2edge_set, entity2edges


def process_data(file_name, entity_dict, relation_dict):
    print('processing %s' % file_name)

    data = []

    file = open(file_name)
    for line in file:
        head, relation, tail = line.strip().split('\t')

        head_idx = entity_dict[head]
        relation_idx = relation_dict[relation]
        tail_idx = entity_dict[tail]

        data.append([head_idx, tail_idx, relation_idx])
    file.close()

    data = np.array(data)

    return data


def get_neighbors(data, edge2entities, edge2relation, entity2edge_set, entity2edges, is_train, args):
    # the first element in edge_list is relations of all triples
    edges_list = [data[:, 2]]
    null_edge = len(edge2entities) - 1

    if is_train:
        for i in range(args.iteration):
            edges_list.append([])

        for edge, (head, tail, _) in enumerate(data):
            # save the two rows temporarily
            row_head = entity2edges[head]
            row_tail = entity2edges[tail]

            # replace the two rows (head and tail) with new sampled neighbors
            entity2edges[head] = get_sampled_neighbors(entity2edge_set[head], edge, null_edge, args.sample)
            entity2edges[tail] = get_sampled_neighbors(entity2edge_set[tail], edge, null_edge, args.sample)

            for j in range(args.iteration):
                if j == 0:
                    neighbor_entities = np.array([head, tail])
                else:
                    neighbor_entities = edge2entities[edges_list[j][edge]].flatten()
                neighbor_edges = entity2edges[neighbor_entities].flatten()
                edges_list[j + 1].append(neighbor_edges)

            entity2edges[head] = row_head
            entity2edges[tail] = row_tail

        for i in range(args.iteration):
            edges_list[i + 1] = np.array(edges_list[i + 1])

    else:
        for i in range(args.iteration):
            if i == 0:
                neighbor_entities = data[:, 0:2]
            else:
                neighbor_entities = edge2entities[edges_list[-1]].reshape([len(data), -1])
            neighbor_edges = entity2edges[neighbor_entities].reshape([len(data), -1])
            edges_list.append(neighbor_edges)

    # map each edge to its relation type
    for i in range(args.iteration):
        edges_list[i + 1] = edge2relation[edges_list[i + 1]]

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


def load_data(args):
    entity_dict = process_entities('../data/' + args.dataset + '/entities.dict')
    relation_dict = process_relations('../data/' + args.dataset + '/relations.dict', args)

    edge2entities, edge2relation, entity2edge_set, entity2edges = process_kg(
        '../data/' + args.dataset + '/train.txt', entity_dict, relation_dict, args.sample)

    train_data = process_data('../data/' + args.dataset + '/train.txt', entity_dict, relation_dict)
    val_data = process_data('../data/' + args.dataset + '/valid.txt', entity_dict, relation_dict)
    test_data = process_data('../data/' + args.dataset + '/test.txt', entity_dict, relation_dict)

    print('sampling neighbor edges')
    train_neighbors = get_neighbors(train_data, edge2entities, edge2relation, entity2edge_set, entity2edges, True, args)
    val_neighbors = get_neighbors(val_data, edge2entities, edge2relation, entity2edge_set, entity2edges, False, args)
    test_neighbors = get_neighbors(test_data, edge2entities, edge2relation, entity2edge_set, entity2edges, False, args)

    return train_neighbors, val_neighbors, test_neighbors, len(relation_dict)
