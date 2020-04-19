import numpy as np
import tensorflow as tf
from aggregators import MeanAggregator, ConcatAggregator, CrossAggregator


class PathCon(object):
    def __init__(self, args, n_relations, params_for_neighbors, params_for_paths):
        self._parse_args(args, n_relations, params_for_neighbors, params_for_paths)
        self._build_inputs()
        self._build_model()
        self._build_train()
        self._build_eval()

    def _parse_args(self, args, n_relations, params_for_neighbors, params_for_paths):
        self.n_relations = n_relations

        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.hidden_dim = args.dim
        self.l2 = args.l2
        self.lr = args.lr
        self.feature_type = args.feature_type

        self.use_context = args.use_context
        if self.use_context:
            self.entity2edges = tf.constant(params_for_neighbors[0], tf.int32, name='entity2edges')
            self.edge2entities = tf.constant(params_for_neighbors[1], tf.int32, name='edge2entities')
            self.edge2relation = tf.constant(params_for_neighbors[2], tf.int32, name='edge2relation')
            self.neighbor_samples = args.neighbor_samples
            self.context_hops = args.context_hops
            if args.neighbor_agg == 'mean':
                self.neighbor_agg = MeanAggregator
            elif args.neighbor_agg == 'concat':
                self.neighbor_agg = ConcatAggregator
            elif args.neighbor_agg == 'cross':
                self.neighbor_agg = CrossAggregator

        self.use_path = args.use_path
        if self.use_path:
            self.path_type = args.path_type
            if self.path_type == 'embedding':
                self.n_paths = params_for_paths[0]
            elif self.path_type == 'rnn':
                self.max_path_len = args.max_path_len
                self.path_samples = args.path_samples
                self.path_agg = args.path_agg
                self.id2path = tf.constant(params_for_paths[0], tf.int32, name='id2path')
                self.id2length = tf.constant(params_for_paths[1], tf.int32, name='id2length')

    def _build_inputs(self):
        if self.use_context:
            self.entity_pairs = tf.placeholder(tf.int32, [self.batch_size, 2], name='entity_pairs')
            self.train_edges = tf.placeholder(tf.int32, [self.batch_size], name='train_edges')

        if self.use_path:
            if self.path_type == 'embedding':
                self.path_features = tf.sparse.placeholder(tf.float64, [self.batch_size, self.n_paths], name='paths')
            elif self.path_type == 'rnn':
                self.path_ids = tf.placeholder(tf.int32, [self.batch_size, self.path_samples], name='paths')

        self.labels = tf.placeholder(tf.int32, [self.batch_size], name='labels')

    def _build_model(self):
        # define initial relation features
        if self.use_context or (self.use_path and self.path_type == 'rnn'):
            self._build_relation_feature()

        self.scores = 0.0

        if self.use_context:
            edges_list, mask_list = self._get_neighbors_and_masks(self.labels, self.entity_pairs, self.train_edges)
            self.aggregators = self._get_neighbor_aggregators()  # define aggregators for each layer
            self.aggregated_neighbors = self._aggregate_neighbors(edges_list, mask_list)  # [batch_size, n_relations]
            self.scores += self.aggregated_neighbors

        if self.use_path:
            if self.path_type == 'embedding':
                self.W, self.b = self._get_weight_and_bias(self.n_paths, self.n_relations)  # [batch_size, n_relations]
                self.scores += tf.sparse_tensor_dense_matmul(self.path_features, self.W) + self.b

            elif self.path_type == 'rnn':
                rnn_output = self._rnn(self.path_ids)  # [batch_size, path_samples, n_relations]
                self.scores += self._aggregate_paths(rnn_output)

        # narrow the range of scores to [0, 1] for the ease of calculating ranking-based metrics
        self.scores_normalized = tf.sigmoid(self.scores)

    def _build_relation_feature(self):
        if self.feature_type == 'id':
            self.relation_dim = self.n_relations
            self.relation_features = tf.eye(self.n_relations, dtype=tf.float64)
        elif self.feature_type == 'bow':
            bow = np.load('../data/' + self.dataset + '/bow.npy')
            self.relation_dim = bow.shape[1]
            self.relation_features = tf.constant(bow, tf.float64)
        elif self.feature_type == 'bert':
            bert = np.load('../data/' + self.dataset + '/bert.npy')
            self.relation_dim = bert.shape[1]
            self.relation_features = tf.constant(bert, tf.float64)

        # the feature of the last relation (the null relation) is a zero vector
        self.relation_features = tf.concat([self.relation_features, tf.zeros([1, self.relation_dim], tf.float64)],
                                           axis=0, name='relation_features')

    def _get_neighbors_and_masks(self, relations, entity_pairs, train_edges):
        edges_list = [relations]
        masks = []
        train_edges = tf.expand_dims(train_edges, -1)  # [batch_size, 1]

        for i in range(self.context_hops):
            if i == 0:
                neighbor_entities = entity_pairs
            else:
                neighbor_entities = tf.reshape(tf.gather(self.edge2entities, edges_list[-1]), [self.batch_size, -1])
            neighbor_edges = tf.reshape(tf.gather(self.entity2edges, neighbor_entities), [self.batch_size, -1])
            edges_list.append(neighbor_edges)

            mask = neighbor_edges - train_edges  # [batch_size, -1]
            mask = tf.cast(tf.cast(mask, tf.bool), tf.float64)  # [batch_size, -1]
            masks.append(mask)
        return edges_list, masks

    def _get_neighbor_aggregators(self):
        aggregators = []  # store all aggregators

        if self.context_hops == 1:
            aggregators.append(self.neighbor_agg(batch_size=self.batch_size,
                                                 input_dim=self.relation_dim,
                                                 output_dim=self.n_relations,
                                                 self_included=False))
        else:
            # the first layer
            aggregators.append(self.neighbor_agg(batch_size=self.batch_size,
                                                 input_dim=self.relation_dim,
                                                 output_dim=self.hidden_dim,
                                                 act=tf.nn.relu))
            # middle layers
            for i in range(self.context_hops - 2):
                aggregators.append(self.neighbor_agg(batch_size=self.batch_size,
                                                     input_dim=self.hidden_dim,
                                                     output_dim=self.hidden_dim,
                                                     act=tf.nn.relu))
            # the last layer
            aggregators.append(self.neighbor_agg(batch_size=self.batch_size,
                                                 input_dim=self.hidden_dim,
                                                 output_dim=self.n_relations,
                                                 self_included=False))
        return aggregators

    def _aggregate_neighbors(self, edge_list, mask_list):
        # translate edges IDs to relations IDs, then to features
        edge_vectors = [tf.nn.embedding_lookup(self.relation_features, edge_list[0])]
        for edges in edge_list[1:]:
            relations = tf.gather(self.edge2relation, edges)
            edge_vectors.append(tf.nn.embedding_lookup(self.relation_features, relations))

        # shape of edge vectors:
        # [[batch_size, relation_dim],
        #  [batch_size, 2 * neighbor_samples, relation_dim],
        #  [batch_size, (2 * neighbor_samples) ^ 2, relation_dim],
        #  ...]

        for i in range(self.context_hops):
            aggregator = self.aggregators[i]
            edge_vectors_next_iter = []
            neighbors_shape = [self.batch_size, -1, 2, self.neighbor_samples, aggregator.input_dim]
            masks_shape = [self.batch_size, -1, 2, self.neighbor_samples, 1]

            for hop in range(self.context_hops - i):
                vector = aggregator(self_vectors=edge_vectors[hop],
                                    neighbor_vectors=tf.reshape(edge_vectors[hop + 1], neighbors_shape),
                                    masks=tf.reshape(mask_list[hop], masks_shape))
                edge_vectors_next_iter.append(vector)
            edge_vectors = edge_vectors_next_iter

        # edge_vectos[0]: [self.batch_size, 1, self.n_relations]
        res = tf.reshape(edge_vectors[0], [self.batch_size, self.n_relations])
        return res

    def _rnn(self, path_ids):
        path_ids = tf.reshape(path_ids, [self.batch_size * self.path_samples])  # [batch_size * path_samples]
        paths = tf.nn.embedding_lookup(self.id2path, path_ids)  # [batch_size * path_samples, max_path_len]
        # [batch_size * path_samples, max_path_len, relation_dim]
        path_features = tf.nn.embedding_lookup(self.relation_features, paths)
        lengths = tf.nn.embedding_lookup(self.id2length, path_ids)  # [batch_size * path_samples]

        cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_dim, name='basic_lstm_cell')
        initial_state = cell.zero_state(self.batch_size * self.path_samples, tf.float64)

        # [batch_size * path_samples, hidden_dim]
        _, last_state = tf.nn.dynamic_rnn(cell, path_features, sequence_length=lengths, initial_state=initial_state)

        self.W, self.b = self._get_weight_and_bias(self.hidden_dim, self.n_relations)
        output = tf.matmul(last_state.h, self.W) + self.b  # [batch_size * path_samples, n_relations]
        output = tf.reshape(output, [self.batch_size, self.path_samples, self.n_relations])

        return output

    def _aggregate_paths(self, inputs):
        # input shape: [batch_size, path_samples, n_relations]

        if self.path_agg == 'mean':
            output = tf.reduce_mean(inputs, axis=1)  # [batch_size, n_relations]
        elif self.path_agg == 'att':
            assert self.use_context
            aggregated_neighbors = tf.expand_dims(self.aggregated_neighbors, axis=1)  # [batch_size, 1, n_relations]
            attention_weights = tf.reduce_sum(aggregated_neighbors * inputs, axis=-1)  # [batch_size, path_samples]
            attention_weights = tf.nn.softmax(attention_weights, axis=-1)  # [batch_size, path_samples]
            attention_weights = tf.expand_dims(attention_weights, axis=-1)  # [batch_size, path_samples, 1]
            output = tf.reduce_sum(attention_weights * inputs, axis=1)  # [batch_size, n_relations]
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
        correct_predictions = tf.equal(self.labels, tf.cast(tf.argmax(self.scores, axis=-1), tf.int32))
        self.acc = tf.reduce_mean(tf.cast(correct_predictions, tf.float64))

    @staticmethod
    def _get_weight_and_bias(input_dim, output_dim):
        weight = tf.get_variable('weight', [input_dim, output_dim], tf.float64, tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable('bias', [output_dim], tf.float64, tf.zeros_initializer())
        return weight, bias

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)

    def eval(self, sess, feed_dict):
        return sess.run([self.acc, self.scores_normalized], feed_dict)
