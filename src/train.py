import tensorflow as tf
import numpy as np
from model import MPNN
from collections import defaultdict


def train(args, data):
    neighbors, paths, n_relations = data
    train_neighbors, valid_neighbors, test_neighbors = neighbors
    train_paths, valid_paths, test_paths = paths

    '''
    all_data = np.concatenate([train_data, valid_data, test_data], axis=0)

    true_relations = defaultdict(list)
    for head, tail, relation in all_data:
        true_relations[(head, tail)].append(relation)
    '''

    model = MPNN(args, n_relations)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(args.epoch):
            # data shuffling
            index = np.arange(len(train_neighbors[0]))
            np.random.shuffle(index)
            for i in range(args.iteration + 1):
                train_neighbors[i] = train_neighbors[i][index]

            # training
            start = 0
            # skip the last incomplete minibatch if its size < batch size
            while start + args.batch_size <= len(train_neighbors[0]):
                _, loss = model.train(
                    sess, get_feed_dict(model, train_neighbors, start, start + args.batch_size, args.iteration))
                start += args.batch_size

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
            train_acc = evaluate(sess, model, train_neighbors, args.batch_size, args.iteration)
            valid_acc = evaluate(sess, model, valid_neighbors, args.batch_size, args.iteration)
            test_acc = evaluate(sess, model, test_neighbors, args.batch_size, args.iteration)
            print('train acc: %.3f   valid acc: %.3f   test acc: %.3f' % (train_acc, valid_acc, test_acc))


def get_feed_dict(model, neighbors, start, end, n_iteration):
    feed_dict = {}
    for i in range(n_iteration + 1):
        feed_dict[model.neighbors_list[i]] = neighbors[i][start:end]
    return feed_dict


def evaluate(sess, model, neighbors, batch_size, n_iteration):
    acc_list = []
    start = 0
    while start + batch_size <= len(neighbors[0]):
        acc = model.eval(sess, get_feed_dict(model, neighbors, start, start + batch_size, n_iteration))
        acc_list.append(acc)
        start += batch_size
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
