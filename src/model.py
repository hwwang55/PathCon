import numpy as np
import tensorflow as tf
from aggregators import MeanAggregator, ConcatAggregator


class MPNN(object):
    def __init__(self, args, n_relations, params_for_paths):
        self._parse_args(args, n_relations, params_for_paths)
        self._build_inputs()
        self._build_model()
        self._build_train()
        self._build_eval()

    def _parse_args(self, args, n_relations, params_for_paths):
        self.n_relations = n_relations

        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.hidden_dim = args.dim
        self.l2_weight = args.l2
        self.lr = args.lr
        self.dropout = args.dropout

        self.use_ls = args.use_ls
        if self.use_ls:
            self.feature_mode = args.feature
            self.n_neighbors = args.sample
            self.n_iterations = args.iteration
            if args.aggregator == 'mean':
                self.aggregator_class = MeanAggregator
            elif args.aggregator == 'concat':
                self.aggregator_class = ConcatAggregator
            elif args.aggregator == 'cross':
                pass
                # TODO

        self.use_e2e = args.use_e2e
        if self.use_e2e:
            self.path_embedding_mode = args.path_embedding
            if self.path_embedding_mode == 'id':
                self.n_paths = params_for_paths[0]
            elif self.path_embedding_mode == 'rnn':
                self.feature_mode = args.feature
                self.max_path_len = args.max_path_len
                self.p_sample = args.p_sample
                self.id2path = tf.constant(params_for_paths[0], dtype=tf.int32, name='id2path')
                self.id2length = tf.constant(params_for_paths[1], dtype=tf.int32, name='id2length')

    def _build_inputs(self):
        if self.use_ls:
            self.neighbors_list = [tf.placeholder(dtype=tf.int32, shape=[self.batch_size], name='neighbors_0')]
            for i in range(self.n_iterations):
                self.neighbors_list.append(tf.placeholder(dtype=tf.int32,
                                                          shape=[self.batch_size, pow(self.n_neighbors * 2, i + 1)],
                                                          name='neighbors_' + str(i + 1)))

        if self.use_e2e:
            if self.path_embedding_mode == 'id':
                self.path_features = tf.sparse.placeholder(dtype=tf.float64,
                                                           shape=[self.batch_size, self.n_paths],
                                                           name='paths')
            elif self.path_embedding_mode == 'rnn':
                self.path_ids = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.p_sample], name='paths')

        self.labels = tf.placeholder(dtype=tf.int32, shape=[self.batch_size], name='labels')

    def _build_model(self):
        # define initial relation features
        if self.use_ls or (self.use_e2e and self.path_embedding_mode == 'rnn'):
            self._build_relation_feature()

        self.scores = 0.0
        #self.scores_normalized = tf.sigmoid(self.scores)  # [batch_size, n_relations]

        if self.use_ls:
            self.aggregators = self._get_aggregators_for_ls()  # define aggregators for each layer
            self.scores += self._aggregate_neighbors()  # [batch_size, n_relations]

        if self.use_e2e:
            if self.path_embedding_mode == 'id':
                # [batch_size, n_labels]
                self.W, self.b = self._get_w_and_b(self.n_paths, self.n_relations)
                self.scores += tf.sparse_tensor_dense_matmul(self.path_features, self.W) + self.b

            elif self.path_embedding_mode == 'rnn':
                # [batch_size, p_sample, hidden_dim]
                rnn_output = self._rnn()

                # [batch_size, hidden_dim]
                aggregated = self._aggreate_paths(rnn_output)

                # [batch_size, n_relations]
                self.W, self.b = self._get_w_and_b(self.hidden_dim, self.n_relations)
                self.scores += tf.matmul(aggregated, self.W) + self.b

    def _build_relation_feature(self):
        # the feature of the last relation (the null relation) is a zero vector
        if self.feature_mode == 'id':
            self.relation_dim = self.n_relations
            self.relation_features = tf.concat([tf.eye(self.n_relations, dtype=tf.float64),
                                                tf.zeros([1, self.relation_dim], dtype=tf.float64)],
                                               axis=0,
                                               name='relation_features')
        elif self.feature_mode == 'bow':
            bow = np.load('../data/' + self.dataset + '/bow.npy')
            self.relation_dim = bow.shape[1]
            self.relation_features = tf.concat([tf.constant(bow, dtype=tf.float64),
                                                tf.zeros([1, self.relation_dim], dtype=tf.float64)],
                                               axis=0,
                                               name='relation_features')
        elif self.feature_mode == 'bert':
            bert = np.load('../data/' + self.dataset + '/' + self.feature_mode + '.npy')
            self.relation_dim = bert.shape[1]
            self.relation_features = tf.concat([tf.constant(bert, dtype=tf.float64),
                                                tf.zeros([1, self.relation_dim], dtype=tf.float64)],
                                               axis=0,
                                               name='relation_features')
        else:
            raise ValueError('unknown relation feature mode')

    def _get_aggregators_for_ls(self):
        aggregators = []  # store all aggregators

        if self.n_iterations == 1:
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
            for i in range(self.n_iterations - 2):
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

    def _aggregate_neighbors(self):
        # translate edges IDs to relations IDs, then to features
        edge_vectors = [tf.nn.embedding_lookup(self.relation_features, edges) for edges in self.neighbors_list]

        for i in range(self.n_iterations):
            aggregator = self.aggregators[i]
            edge_vectors_next_iter = []
            for hop in range(self.n_iterations - i):
                vector = aggregator(self_vectors=edge_vectors[hop],
                                    neighbor_vectors=tf.reshape(
                                        edge_vectors[hop + 1],
                                        [self.batch_size, -1, 2, self.n_neighbors, aggregator.input_dim]))
                edge_vectors_next_iter.append(vector)
            edge_vectors = edge_vectors_next_iter

        # edge_vectos[0]: [self.batch_size, 1, self.n_relations]
        res = tf.reshape(edge_vectors[0], [self.batch_size, self.n_relations])
        return res

    def _rnn(self):
        # [batch_size * p_sample]
        path_ids = tf.reshape(self.path_ids, [self.batch_size * self.p_sample])

        # [batch_size * p_sample, max_path_len]
        paths = tf.nn.embedding_lookup(self.id2path, path_ids)

        # [batch_size * p_sample, max_path_len, relation_dim]
        path_features = tf.nn.embedding_lookup(self.relation_features, paths)

        # [batch_size * p_sample]
        path_lengths = tf.nn.embedding_lookup(self.id2length, path_ids)

        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.hidden_dim)
        _, last_state = tf.nn.dynamic_rnn(cell,
                                          path_features,
                                          sequence_length=path_lengths,
                                          initial_state=cell.zero_state(self.batch_size * self.p_sample, tf.float64))

        # [batch_size, p_sample, hidden_dim]
        last_state = tf.reshape(last_state.h, [self.batch_size, self.p_sample, self.hidden_dim])
        return last_state

    def _aggreate_paths(self, inputs):
        # TODO: other aggregation method
        output = tf.reduce_mean(inputs, axis=1)
        return output

    def _build_train(self):
        self.base_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels,
                                                                                       logits=self.scores))
        self.l2_loss = 0.0

        if self.use_ls:
            for aggregator in self.aggregators:
                self.l2_loss += self.l2_weight * tf.nn.l2_loss(aggregator.weights)  # l2 loss of each aggregator

        if self.use_e2e:
            if self.path_embedding_mode == 'id':
                self.l2_loss += self.l2_weight * tf.nn.l2_loss(self.W)
            elif self.path_embedding_mode == 'rnn':
                pass

        self.loss = self.base_loss + self.l2_loss
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def _build_eval(self):
        correct_preds = tf.equal(self.labels, tf.cast(tf.argmax(self.scores, axis=-1), tf.int32))
        self.acc = tf.reduce_mean(tf.cast(correct_preds, tf.float64))

    @staticmethod
    def _get_w_and_b(input_dim, output_dim):
        w = tf.get_variable(shape=[input_dim, output_dim],
                            initializer=tf.contrib.layers.xavier_initializer(),
                            dtype=tf.float64,
                            name='W')
        b = tf.get_variable(shape=[output_dim],
                            initializer=tf.zeros_initializer(),
                            dtype=tf.float64,
                            name='b')
        return w, b

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)

    def eval(self, sess, feed_dict):
        #return sess.run(self.scores_normalized, feed_dict)
        return sess.run(self.acc, feed_dict)
