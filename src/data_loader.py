import numpy as np
from collections import defaultdict


def read_dict(file_name):
    print('reading %s' % file_name)
    d = {}
    for line in open(file_name):
        index, name = line.strip().split('\t')
        d[name] = int(index)
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
    for edge_index, line in enumerate(file):
        head, relation, tail = line.strip().split('\t')

        head_idx = entity_dict[head]
        relation_idx = relation_dict[relation]
        tail_idx = entity_dict[tail]

        entity2edge_set[head_idx].add(edge_index)
        entity2edge_set[tail_idx].add(edge_index)
        edge2entities.append([head_idx, tail_idx])
        edge2relation.append(relation_idx)
    file.close()

    # each row in entity2edges is the sampled edges connecting to this entity
    entity2edges = []
    for i in range(len(entity2edge_set)):
        sampled_neighbors = np.random.choice(list(entity2edge_set[i]), size=sampling_size,
                                             replace=len(entity2edge_set[i]) < sampling_size)
        entity2edges.append(sampled_neighbors)

    edge2entities = np.array(edge2entities)
    entity2edges = np.array(entity2edges)
    edge2relation = np.array(edge2relation)

    return edge2entities, entity2edges, edge2relation


def process_data(file_name, entity_dict, relation_dict):
    print('processing %s' % file_name)

    data = []

    file = open(file_name)
    for line in file:
        head, relation, tail = line.strip().split('\t')

        head_idx = entity_dict[head]
        relation_idx = relation_dict[relation]
        tail_idx = entity_dict[tail]

        data.append([head_idx, relation_idx, tail_idx])
    file.close()

    data = np.array(data)

    return data


def load_data(args):
    entity_dict = read_dict('../data/' + args.dataset + '/entities.dict')
    relation_dict = read_dict('../data/' + args.dataset + '/relations.dict')

    edge2entities, entity2edges, edge2relation = process_kg('../data/' + args.dataset + '/train.txt',
                                                            entity_dict, relation_dict, args.sampling_size)

    train_data = process_data('../data/' + args.dataset + '/train.txt', entity_dict, relation_dict)
    val_data = process_data('../data/' + args.dataset + '/valid.txt', entity_dict, relation_dict)
    test_data = process_data('../data/' + args.dataset + '/test.txt', entity_dict, relation_dict)

    return edge2entities, entity2edges, edge2relation, train_data, val_data, test_data
