import numpy as np
import tensorflow as tf


class RelationAgg(object):
    def __init__(self, args, n_relations):
        self._parse_args(args, n_relations)
        self._build_inputs()
        self._build_model()

    def _parse_args(self, args, n_relations):
        self.dataset = args.dataset
        self.hidden_dim = args.dim
        self.iteration = args.iteration
        self.l2 = args.l2
        self.lr = args.lr
        self.rel_feature_mode = args.rel_feature_mode
        self.fine_tune = args.fine_tune
        self.n_relations = n_relations

    def _build_inputs(self):
        self.head_indices = tf.placeholder(dtype=tf.int32, shape=[None], name='head_indices')
        self.tail_indices = tf.placeholder(dtype=tf.int32, shape=[None], name='tail_indices')
        self.relation_indices = tf.placeholder(dtype=tf.int32, shape=[None], name='relation_indices')

    def _build_model(self):
        if self.rel_feature_mode == 'id':
            self.rel_feature_matrix = tf.eye(self.n_relations, dtype=tf.float64, name='rel_features')
            self.rel_feature_dim = self.n_relations
        elif self.rel_feature_mode == 'bow':
            bow = np.load('../data/' + self.dataset + '/bow.npy')
            self.rel_feature_matrix = tf.constant(bow, dtype=tf.float64, name='rel_features')
            self.rel_feature_dim = bow.shape[1]
        elif self.rel_feature_mode == 'bert-small' or 'bert-large':
            bert = np.load('../data/' + self.dataset + '/' + self.rel_feature_mode + '.npy')
            self.rel_feature_matrix = tf.Variable(tf.constant(bert), dtype=tf.float64, trainable=self.fine_tune,
                                                  name='rel_features')
            self.rel_feature_dim = bert.shape[1]
        else:
            raise ValueError('relation feature mode not supported')


