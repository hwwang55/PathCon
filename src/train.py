import tensorflow as tf
import numpy as np
from model import RelationAgg


def train(args, data):
    edge2entities, entity2edges, edge2relation, train_data, val_data, test_data, n_relations = data
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


def get_feed_dict(model, data, start, end):
    feed_dict = {model.entity_pairs: data[start:end, 0:2],
                 model.relations: data[start:end, 2]}
    return feed_dict
