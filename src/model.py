import numpy as np
import tensorflow as tf
from aggregators import MeanAggregator, ConcatAggregator


class MPNet(object):
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
        self.l2 = args.l2
        self.lr = args.lr
        self.dropout = args.dropout

        self.use_gnn = args.use_gnn
        if self.use_gnn:
            self.feature_mode = args.feature_mode
            self.neighbor_samples = args.neighbor_samples
            self.gnn_layers = args.gnn_layers
            if args.neighbor_agg == 'mean':
                self.neighbor_agg = MeanAggregator
            elif args.neighbor_agg == 'concat':
                self.neighbor_agg = ConcatAggregator
            elif args.neighbor_agg == 'cross':
                pass  # TODO
            else:
                raise ValueError('unknown neighbor_agg')

        self.use_path = args.use_path
        if self.use_path:
            self.path_mode = args.path_mode
            self.n_paths = params_for_paths[0]
            if self.path_mode == 'rnn':
                self.feature_mode = args.feature_mode
                self.max_path_len = args.max_path_len
                self.path_samples = args.path_samples
                self.path_agg = args.path_agg
                self.id2path = tf.constant(params_for_paths[1], dtype=tf.int32, name='id2path')
                self.id2length = tf.constant(params_for_paths[2], dtype=tf.int32, name='id2length')

    def _build_inputs(self):
        if self.use_gnn:
            self.neighbors_list = [tf.placeholder(dtype=tf.int32, shape=[self.batch_size], name='neighbors_0')]
            for i in range(self.gnn_layers):
                self.neighbors_list.append(tf.placeholder(dtype=tf.int32,
                                                          shape=[self.batch_size, pow(self.neighbor_samples*2, i+1)],
                                                          name='neighbors_' + str(i + 1)))

        if self.use_path:
            if self.path_mode == 'id':
                self.path_features = tf.sparse.placeholder(dtype=tf.float64,
                                                           shape=[self.batch_size, self.n_paths],
                                                           name='paths')
            elif self.path_mode == 'rnn':
                self.path_ids = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.path_samples], name='paths')

        self.labels = tf.placeholder(dtype=tf.int32, shape=[self.batch_size], name='labels')

    def _build_model(self):
        # define initial relation features
        if self.use_gnn or (self.use_path and self.path_mode == 'rnn'):
            self._build_relation_feature()

        self.scores = 0.0
        #self.scores_normalized = tf.sigmoid(self.scores)  # [batch_size, n_relations]

        if self.use_gnn:
            self.aggregators = self._get_neighbor_aggregators()  # define aggregators for each layer
            self.aggregated_neighbors = self._aggregate_neighbors()  # [batch_size, n_relations]
            self.scores += self.aggregated_neighbors

        if self.use_path:
            if self.path_mode == 'id':
                self.W, self.b = self._get_weight_and_bias(self.n_paths, self.n_relations)  # [batch_size, n_relations]
                self.scores += tf.sparse_tensor_dense_matmul(self.path_features, self.W) + self.b

            elif self.path_mode == 'rnn':
                rnn_output = self._rnn()  # [batch_size, path_samples, n_relations]
                self.scores += self._aggregate_paths(rnn_output)

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
            raise ValueError('unknown feature_mode')

    def _get_neighbor_aggregators(self):
        aggregators = []  # store all aggregators

        if self.gnn_layers == 1:
            aggregators.append(self.neighbor_agg(batch_size=self.batch_size,
                                                 input_dim=self.relation_dim,
                                                 output_dim=self.n_relations,
                                                 dropout=self.dropout,
                                                 self_included=False))
        else:
            # the first layer
            aggregators.append(self.neighbor_agg(batch_size=self.batch_size,
                                                 input_dim=self.relation_dim,
                                                 output_dim=self.hidden_dim,
                                                 dropout=self.dropout,
                                                 act=tf.nn.relu))
            # middle layers
            for i in range(self.gnn_layers - 2):
                aggregators.append(self.neighbor_agg(batch_size=self.batch_size,
                                                     input_dim=self.hidden_dim,
                                                     output_dim=self.hidden_dim,
                                                     dropout=self.dropout,
                                                     act=tf.nn.relu))
            # the last layer
            aggregators.append(self.neighbor_agg(batch_size=self.batch_size,
                                                 input_dim=self.hidden_dim,
                                                 output_dim=self.n_relations,
                                                 dropout=self.dropout,
                                                 self_included=False))
        return aggregators

    def _aggregate_neighbors(self):
        # translate edges IDs to relations IDs, then to features
        edge_vectors = [tf.nn.embedding_lookup(self.relation_features, edges) for edges in self.neighbors_list]

        for i in range(self.gnn_layers):
            aggregator = self.aggregators[i]
            edge_vectors_next_iter = []
            for hop in range(self.gnn_layers - i):
                vector = aggregator(self_vectors=edge_vectors[hop],
                                    neighbor_vectors=tf.reshape(
                                        edge_vectors[hop + 1],
                                        [self.batch_size, -1, 2, self.neighbor_samples, aggregator.input_dim]))
                edge_vectors_next_iter.append(vector)
            edge_vectors = edge_vectors_next_iter

        # shape of edge_vectors[0]: [batch_size, 1, n_relations]
        res = tf.reshape(edge_vectors[0], [self.batch_size, self.n_relations])
        return res

    def _rnn(self):
        path_ids = tf.reshape(self.path_ids, [self.batch_size * self.path_samples])  # [batch_size * path_samples]
        paths = tf.nn.embedding_lookup(self.id2path, path_ids)  # [batch_size * path_samples, max_path_len]
        # [batch_size * path_samples, max_path_len, relation_dim]
        path_features = tf.nn.embedding_lookup(self.relation_features, paths)
        path_lengths = tf.nn.embedding_lookup(self.id2length, path_ids)  # [batch_size * path_samples]

        cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_dim, name='basic_lstm_cell')
        initial_state = cell.zero_state(self.batch_size * self.path_samples, tf.float64)

        # [batch_size * path_samples, hidden_dim]
        _, last_state = tf.nn.dynamic_rnn(cell,
                                          path_features,
                                          sequence_length=path_lengths,
                                          initial_state=initial_state)

        self.W, self.b = self._get_weight_and_bias(self.hidden_dim, self.n_relations)
        output = tf.matmul(last_state.h, self.W) + self.b  # [batch_size * path_samples, n_relations]
        output = tf.reshape(output, [self.batch_size, self.path_samples, self.n_relations])

        return output

    def _aggregate_paths(self, inputs):
        # input shape: [batch_size, path_samples, n_relations]

        if self.path_agg == 'avg':
            # [batch_size, n_relations]
            output = tf.reduce_mean(inputs, axis=1)

        elif self.path_agg == 'w_avg':
            self.path_weights = tf.get_variable(shape=[self.n_paths],
                                                initializer=tf.contrib.layers.xavier_initializer(),
                                                dtype=tf.float64,
                                                name='path_weights')
            path_weights_batch = tf.nn.embedding_lookup(self.path_weights, self.path_ids)  # [batch_size, path_samples]
            output = self._weighted_average(inputs, path_weights_batch)

        elif self.path_agg == 'att':
            assert self.use_gnn
            aggregated_neihbors = tf.expand_dims(self.aggregated_neighbors, axis=1)  # [batch_size, 1, n_relations]
            attention_weights = tf.reduce_sum(aggregated_neihbors * inputs, axis=-1)  # [batch_size, path_samples]
            output = self._weighted_average(inputs, attention_weights)

        else:
            raise ValueError('unknown path_agg')

        return output

    def _build_train(self):
        self.base_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels,
                                                                                       logits=self.scores))
        self.l2_loss = self.l2 * sum(tf.nn.l2_loss(var) for var in tf.trainable_variables() if 'bias' not in var.name)
        self.loss = self.base_loss + self.l2_loss
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def _build_eval(self):
        correct_preds = tf.equal(self.labels, tf.cast(tf.argmax(self.scores, axis=-1), tf.int32))
        self.acc = tf.reduce_mean(tf.cast(correct_preds, tf.float64))

    @staticmethod
    def _get_weight_and_bias(input_dim, output_dim):
        weight = tf.get_variable(shape=[input_dim, output_dim],
                                 initializer=tf.contrib.layers.xavier_initializer(),
                                 dtype=tf.float64,
                                 name='weight')
        bias = tf.get_variable(shape=[output_dim],
                               initializer=tf.zeros_initializer(),
                               dtype=tf.float64,
                               name='bias')
        return weight, bias

    @staticmethod
    def _weighted_average(inputs, weights):
        # shape of inputs: [batch_size, path_samples, n_relations]
        # shape of weights: [batch_size, path_samples]
        weights = tf.nn.softmax(weights, axis=-1)  # [batch_size, path_samples]
        weights = tf.expand_dims(weights, axis=-1)  # [batch_size, path_samples, 1]
        output = tf.reduce_sum(weights * inputs, axis=1)  # [batch_size, n_relations]
        return output

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)

    def eval(self, sess, feed_dict):
        #return sess.run(self.scores_normalized, feed_dict)
        return sess.run(self.acc, feed_dict)
