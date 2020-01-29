import tensorflow as tf
import numpy as np
from model import MPNet
from collections import defaultdict
from utils import sparse_to_tuple


args = None


def train(model_args, data):
    global args, model, sess
    args = model_args
    triplets, neighbors, paths, labels, n_relations, params_for_paths = data
    train_triplets, valid_triplets, test_triplets = triplets
    train_neighbors, valid_neighbors, test_neighbors = neighbors
    train_paths, valid_paths, test_paths = paths
    train_labels, valid_labels, test_labels = labels

    show_ranking = True

    true_relations = defaultdict(set)
    if show_ranking:
        for head, tail, relation in train_triplets + valid_triplets + test_triplets:
            true_relations[(head, tail)].add(relation)

    model = MPNet(args, n_relations, params_for_paths)

    with tf.Session() as sess:
        print('start training ...')
        sess.run(tf.global_variables_initializer())

        for step in range(args.epoch):

            # data shuffling
            index = np.arange(len(train_labels))
            np.random.shuffle(index)
            if args.use_gnn:
                for i in range(args.gnn_layers + 1):
                    train_neighbors[i] = train_neighbors[i][index]
            if args.use_path:
                train_paths = train_paths[index]
            train_labels = train_labels[index]

            # training
            start = 0
            while start + args.batch_size <= len(train_labels):
                _, loss = model.train(
                    sess, get_feed_dict(train_neighbors, train_paths, train_labels, start, start + args.batch_size))
                start += args.batch_size

            # evaluation
            print('epoch %2d   ' % step, end='')
            train_acc = evaluate(train_neighbors, train_paths, train_labels, False)
            valid_acc = evaluate(valid_neighbors, valid_paths, valid_labels, False)
            test_acc, test_scores = evaluate(test_neighbors, test_paths, test_labels, show_ranking)

            print('train acc: %.3f   valid acc: %.3f   test acc: %.3f' % (train_acc, valid_acc, test_acc))
            if show_ranking:
                mrr, mr, hit1, hit3, hit5 = calculate_ranking_metrics(test_triplets, test_scores, true_relations)
                print('           mrr: %.3f   mr: %.3f   h1: %.3f   h3: %.3f   h5: %.3f' % (mrr, mr, hit1, hit3, hit5))
                print()


def get_feed_dict(neighbors, paths, labels, start, end):
    feed_dict = {}

    if args.use_gnn:
        for i in range(args.gnn_layers + 1):
            feed_dict[model.neighbors_list[i]] = neighbors[i][start:end]

    if args.use_path:
        if args.path_mode == 'id':
            feed_dict[model.path_features] = sparse_to_tuple(paths[start:end])
        elif args.path_mode == 'rnn':
            feed_dict[model.path_ids] = paths[start:end]

    feed_dict[model.labels] = labels[start:end]

    return feed_dict


def evaluate(neighbors, paths, labels, show_ranking):
    acc_list = []
    scores_list = []

    start = 0
    while start + args.batch_size <= len(labels):
        acc, scores = model.eval(sess, get_feed_dict(neighbors, paths, labels, start, start + args.batch_size))
        acc_list.append(acc)
        if show_ranking:
            scores_list.extend(scores)
        start += args.batch_size

    if show_ranking:
        return float(np.mean(acc_list)), np.array(scores_list)
    else:
        return float(np.mean(acc_list))


def calculate_ranking_metrics(triplets, scores, true_relations):
    for i in range(scores.shape[0]):
        head, tail, relation = triplets[i]
        for j in true_relations[head, tail] - {relation}:
            scores[i, j] -= 1.0

    sorted_indices = np.argsort(-scores, axis=1)
    relations = np.array(triplets)[0:scores.shape[0], 2]
    sorted_indices -= np.expand_dims(relations, 1)
    zero_coordinates = np.argwhere(sorted_indices == 0)
    rankings = zero_coordinates[:, 1] + 1

    mrr = float(np.mean(1 / rankings))
    mr = float(np.mean(rankings))
    hit1 = float(np.mean(rankings <= 1))
    hit3 = float(np.mean(rankings <= 3))
    hit5 = float(np.mean(rankings <= 5))

    return mrr, mr, hit1, hit3, hit5
