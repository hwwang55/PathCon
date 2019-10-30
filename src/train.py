import tensorflow as tf
import numpy as np
from model import RelationAgg
from collections import defaultdict


def train(args, data):
    edge2entities, entity2edges, edge2relation, train_data, val_data, test_data, n_relations = data
    all_data = np.concatenate([train_data, val_data, test_data], axis=0)

    true_relations = defaultdict(list)
    for head, tail, relation in all_data:
        true_relations[(head, tail)].append(relation)

    model = RelationAgg(args, edge2entities, entity2edges, edge2relation, n_relations)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for step in range(args.epoch):
            # training
            np.random.shuffle(train_data)
            start = 0
            # skip the last incomplete minibatch if its size < batch size
            while start + args.batch_size <= train_data.shape[0]:
                _, loss = model.train(sess, get_feed_dict(model, train_data, start, start + args.batch_size))
                start += args.batch_size

            # evaluation
            data = {'train': train_data, 'val': val_data, 'test': test_data}
            print('epoch %d' % step)
            for key, value in data.items():
                mrr, mr, hit1, hit3, hit10 = relation_prediction(sess, model, value, args.batch_size, true_relations)
                print('%s MRR:%.3f  MR:%.3f  HIT@1:%.3f  HIT@3:%.3f  HIT@10:%.3f' % (key, mrr, mr, hit1, hit3, hit10))
            print()


def get_feed_dict(model, data, start, end):
    feed_dict = {model.entity_pairs: data[start:end, 0:2],
                 model.relations: data[start:end, 2]}
    return feed_dict


def relation_prediction(sess, model, data, batch_size, true_relations):
    mrr = []
    mr = []
    hit1 = []
    hit3 = []
    hit10 = []
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
            hit10.append(1.0 if ranking <= 10 else 0.0)
        start += batch_size

    return float(np.mean(mrr)), float(np.mean(mr)), float(np.mean(hit1)), float(np.mean(hit3)), float(np.mean(hit10))
