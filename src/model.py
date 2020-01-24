import numpy as np
import tensorflow as tf
from aggregators import MeanAggregator, ConcatAggregator


class MPNN(object):
    def __init__(self, args, n_relations):
        self._parse_args(args, n_relations)
        self._build_inputs()
        self._build_model()
        self._build_train()
        self._build_eval()

    def _parse_args(self, args, n_relations):
        self.n_relations = n_relations
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.n_neighbors = args.sample
        self.hidden_dim = args.dim
        self.iteration = args.iteration
        self.l2_weight = args.l2
        self.lr = args.lr
        self.dropout = args.dropout
        self.relation_feature_mode = args.feature

        if args.aggregator == 'mean':
            self.aggregator_class = MeanAggregator
        elif args.aggregator == 'concat':
            self.aggregator_class = ConcatAggregator
        else:
            raise ValueError('unknown aggregator')

    def _build_inputs(self):
        self.neighbors_list = [tf.placeholder(dtype=tf.int32, shape=[self.batch_size], name='neighbors_0')]

        for i in range(self.iteration):
            self.neighbors_list.append(tf.placeholder(dtype=tf.int32,
                                                      shape=[self.batch_size, pow(self.n_neighbors * 2, i + 1)],
                                                      name='neighbors_' + str(i + 1)))

    def _build_model(self):
        # prepare initial relation features
        # the feature of the last relation (the null relation) is a zero vector
        if self.relation_feature_mode == 'id':
            self.relation_dim = self.n_relations
            self.relation_features = tf.concat([tf.eye(self.n_relations, dtype=tf.float32),
                                                tf.zeros([1, self.relation_dim], dtype=tf.float32)],
                                               axis=0,
                                               name='relation_features')
        elif self.relation_feature_mode == 'bow':
            bow = np.load('../data/' + self.dataset + '/bow.npy')
            self.relation_dim = bow.shape[1]
            self.relation_features = tf.concat([tf.constant(bow, dtype=tf.float32),
                                                tf.zeros([1, self.relation_dim], dtype=tf.float32)],
                                               axis=0,
                                               name='relation_features')
        elif self.relation_feature_mode == 'bert-small' or 'bert-large':
            bert = np.load('../data/' + self.dataset + '/' + self.relation_feature_mode + '.npy')
            self.relation_dim = bert.shape[1]
            self.relation_features = tf.concat([tf.constant(bert, dtype=tf.float32),
                                                tf.zeros([1, self.relation_dim], dtype=tf.float32)],
                                               axis=0,
                                               name='relation_features')
        else:
            raise ValueError('unknown relation feature mode')

        # define aggregators for each layer
        self.aggregators = self._get_aggregators()

        # aggregate neighbors
        self.scores = self._aggregate()  # [batch_size, n_relations]
        #self.scores_normalized = tf.sigmoid(self.scores)  # [batch_size, n_relations]

    def _get_aggregators(self):
        aggregators = []  # store all aggregators

        if self.iteration == 1:
            aggregators.append(self.aggregator_class(batch_size=self.batch_size,
                                                     input_dim=self.relation_dim,
                                                     output_dim=self.n_relations,
                                                     dropout=self.dropout,
                                                     self_included=False))
        else:
            # the first layer
            aggregators.append(self.aggregator_class(batch_size=self.batch_size,
                                                     input_dim=self.relation_dim,
                                                     output_dim=self.hidden_dim,
                                                     dropout=self.dropout,
                                                     act=tf.nn.relu))
            # middle layers
            for i in range(self.iteration - 2):
                aggregators.append(self.aggregator_class(batch_size=self.batch_size,
                                                         input_dim=self.hidden_dim,
                                                         output_dim=self.hidden_dim,
                                                         dropout=self.dropout,
                                                         act=tf.nn.relu))
            # the last layer
            aggregators.append(self.aggregator_class(batch_size=self.batch_size,
                                                     input_dim=self.hidden_dim,
                                                     output_dim=self.n_relations,
                                                     dropout=self.dropout,
                                                     self_included=False))
        return aggregators

    def _aggregate(self):
        # translate edges IDs to relations IDs, then to features
        edge_vectors = [tf.nn.embedding_lookup(self.relation_features, edges) for edges in self.neighbors_list]

        for i in range(self.iteration):
            aggregator = self.aggregators[i]
            edge_vectors_next_iter = []
            for hop in range(self.iteration - i):
                vector = aggregator(self_vectors=edge_vectors[hop],
                                    neighbor_vectors=tf.reshape(
                                        edge_vectors[hop + 1],
                                        [self.batch_size, -1, 2, self.n_neighbors, aggregator.input_dim]))
                edge_vectors_next_iter.append(vector)
            edge_vectors = edge_vectors_next_iter

        # edge_vectos[0]: [self.batch_size, 1, self.n_relations]
        res = tf.reshape(edge_vectors[0], [self.batch_size, self.n_relations])
        return res

    def _build_train(self):
        self.base_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.neighbors_list[0],
                                                                                       logits=self.scores))
        self.l2_loss = 0.0
        for aggregator in self.aggregators:
            self.l2_loss += self.l2_weight * tf.nn.l2_loss(aggregator.weights)  # l2 loss of each aggregator
        self.loss = self.base_loss + self.l2_loss

        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def _build_eval(self):
        correct_preds = tf.equal(self.neighbors_list[0], tf.cast(tf.argmax(self.scores, axis=-1), tf.int32))
        self.acc = tf.reduce_mean(tf.cast(correct_preds, tf.float64))

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)

    def eval(self, sess, feed_dict):
        #return sess.run(self.scores_normalized, feed_dict)
        return sess.run(self.acc, feed_dict)
