import tensorflow as tf
import numpy as np
from model import MPNN
from collections import defaultdict
from utils import sparse_to_tuple


args = None


def train(model_args, data):
    global args, model, sess
    args = model_args
    neighbors, paths, labels, n_relations, params_for_paths = data
    train_neighbors, valid_neighbors, test_neighbors = neighbors
    train_paths, valid_paths, test_paths = paths
    train_labels, valid_labels, test_labels = labels

    '''
    all_data = np.concatenate([train_data, valid_data, test_data], axis=0)

    true_relations = defaultdict(list)
    for head, tail, relation in all_data:
        true_relations[(head, tail)].append(relation)
    '''

    model = MPNN(args, n_relations, params_for_paths)

    with tf.Session() as sess:
        print('start training ...')
        sess.run(tf.global_variables_initializer())

        for step in range(args.epoch):
            # data shuffling
            index = np.arange(len(train_labels))
            np.random.shuffle(index)
            if args.use_ls:
                for i in range(args.iteration + 1):
                    train_neighbors[i] = train_neighbors[i][index]
            if args.use_e2e:
                train_paths = train_paths[index]
            train_labels = train_labels[index]

            # training
            s = 0
            while s + args.batch_size <= len(train_labels):
                _, loss = model.train(
                    sess, get_feed_dict(train_neighbors, train_paths, train_labels, s, s + args.batch_size))
                s += args.batch_size

            # evaluation
            '''
            data = {'train': train_data, 'valid': val_data, 'test': test_data}
            print('epoch %d' % step)
            for key, value in data.items():
                mrr, mr, hit1, hit3, hit5 = relation_prediction(sess, model, value, args.batch_size, true_relations)
                print('%s\tmrr:%.3f  mr:%.3f  h@1:%.3f  h@3:%.3f  h@5:%.3f' % (key, mrr, mr, hit1, hit3, hit5))
            print()
            '''
            print('epoch %d   ' % step, end='')
            train_acc = evaluate(train_neighbors, train_paths, train_labels)
            valid_acc = evaluate(valid_neighbors, valid_paths, valid_labels)
            test_acc = evaluate(test_neighbors, test_paths, test_labels)
            print('train acc: %.3f   valid acc: %.3f   test acc: %.3f' % (train_acc, valid_acc, test_acc))


def get_feed_dict(neighbors, paths, labels, start, end):
    feed_dict = {}

    if args.use_ls:
        for i in range(args.iteration + 1):
            feed_dict[model.neighbors_list[i]] = neighbors[i][start:end]

    if args.use_e2e:
        if args.path_embedding == 'id':
            feed_dict[model.path_features] = sparse_to_tuple(paths[start:end])
        elif args.path_embedding == 'rnn':
            feed_dict[model.path_ids] = paths[start:end]

    feed_dict[model.labels] = labels[start:end]

    return feed_dict


def evaluate(neighbors, paths, labels):
    acc_list = []
    start = 0
    while start + args.batch_size <= len(labels):
        acc = model.eval(sess, get_feed_dict(neighbors, paths, labels, start, start + args.batch_size))
        acc_list.append(acc)
        start += args.batch_size
    return float(np.mean(acc_list))


'''
def relation_prediction(sess, model, data, batch_size, true_relations):
    mrr = []
    mr = []
    hit1 = []
    hit3 = []
    hit5 = []
    start = 0
    while start + batch_size <= data.shape[0]:
        scores = model.eval(sess, get_feed_dict(model, data, start, start + batch_size))
        for i in range(batch_size):
            head, tail, relation = data[i]
            for j in true_relations[head, tail]:
                if j != relation:
                    scores[i, j] -= 1.0
            sorted_indices = np.argsort(-scores[i])

            ranking = 0
            for j in range(len(sorted_indices)):
                if sorted_indices[j] == relation:
                    ranking = j + 1
                    break
            mrr.append(1.0 / ranking)
            mr.append(float(ranking))
            hit1.append(1.0 if ranking <= 1 else 0.0)
            hit3.append(1.0 if ranking <= 3 else 0.0)
            hit5.append(1.0 if ranking <= 5 else 0.0)
        start += batch_size

    return float(np.mean(mrr)), float(np.mean(mr)), float(np.mean(hit1)), float(np.mean(hit3)), float(np.mean(hit5))
'''
