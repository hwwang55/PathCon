import numpy as np
import tensorflow as tf
from layers import Dense
from aggregators import MeanAggregator, ConcatAggregator


class RelationAgg(object):
    def __init__(self, args, edge2entities, entity2edges, edge2relation, n_relations):
        self._parse_args(args, edge2entities, entity2edges, edge2relation, n_relations)
        self._build_inputs()
        self._build_model()
        self._build_train()

    def _parse_args(self, args, edge2entities, entity2edges, edge2relation, n_relations):
        self.edge2entities = edge2entities
        self.entity2edges = entity2edges
        self.edge2relation = edge2relation
        self.n_relations = n_relations

        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.n_neighbor = args.sample
        self.dim = args.dim
        self.iteration = args.iteration
        self.l2_weight = args.l2
        self.lr = args.lr
        self.dropout = args.dropout
        self.relation_feature_mode = args.feature
        self.fine_tune = args.finetune

        if args.aggregator == 'mean':
            self.aggregator_class = MeanAggregator
        elif args.aggregator == 'concat':
            self.aggregator_class = ConcatAggregator
        else:
            raise ValueError('unknown aggregator')

    def _build_inputs(self):
        self.entity_pairs = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, 2], name='entity_pairs')
        self.relations = tf.placeholder(dtype=tf.int32, shape=[self.batch_size], name='relations')

    def _build_model(self):
        if self.relation_feature_mode == 'id':
            self.relation_features = tf.eye(self.n_relations, dtype=tf.float32, name='relation_features')
            self.relation_dim = self.n_relations
        elif self.relation_feature_mode == 'bow':
            bow = np.load('../data/' + self.dataset + '/bow.npy')
            self.relation_features = tf.constant(bow, dtype=tf.float32, name='relation_features')
            self.relation_dim = bow.shape[1]
        elif self.relation_feature_mode == 'bert-small' or 'bert-large':
            bert = np.load('../data/' + self.dataset + '/' + self.relation_feature_mode + '.npy')
            self.relation_features = tf.Variable(tf.constant(bert), dtype=tf.float32, trainable=self.fine_tune,
                                                 name='relation_features')
            self.relation_dim = bert.shape[1]
        else:
            raise ValueError('unknown relation feature mode')

        self.emb_mapping = Dense(self.relation_dim, self.dim, dropout=self.dropout, name='embedding_mapping')
        self.relation_embeddings = self.emb_mapping(self.relation_features)  # [n_relations, dim]

        edges_list = self._get_neighbors(self.relations, self.entity_pairs)
        self.aggregators = self._get_aggregators()
        pred_relation_emb = self._aggregate(edges_list)  # [batch_size, dim]
        pred_relation_emb = tf.expand_dims(pred_relation_emb, 1)  # [batch_size, 1, dim]
        self.scores = tf.reduce_sum(pred_relation_emb * self.relation_embeddings, axis=-1)  # [batch_size, n_relations]
        self.scores_normalized = tf.sigmoid(self.scores)  # [batch_size, n_relations]

    def _get_neighbors(self, relations, entities):
        edges_list = [relations]  # the 'relations' servers as a placeholder only
        for i in range(self.iteration):
            if i == 0:
                neighbor_entities = entities
            else:
                neighbor_entities = tf.reshape(tf.gather(self.edge2entities, edges_list[-1]), [self.batch_size, -1])
            neighbor_edges = tf.reshape(tf.gather(self.entity2edges, neighbor_entities), [self.batch_size, -1])
            edges_list.append(neighbor_edges)
        return edges_list

    def _get_aggregators(self):
        aggregators = []  # store all aggregators

        # non-last layers
        for i in range(self.iteration - 1):
            aggregator = self.aggregator_class(self.batch_size, self.dim, self.dropout, act=tf.nn.relu)
            aggregators.append(aggregator)

        # the last layer
        # self_vectors are not included because we do not want to aggregate the relation which needs to be predicted
        aggregator = self.aggregator_class(self.batch_size, self.dim, self.dropout, act=tf.nn.tanh, self_included=False)
        aggregators.append(aggregator)

        return aggregators

    def _aggregate(self, edge_list):
        edge_vectors = [tf.nn.embedding_lookup(self.relation_embeddings, edge_list[0])]
        for edges in edge_list[1:]:
            relations = tf.gather(self.edge2relation, edges)
            edge_vectors.append(tf.nn.embedding_lookup(self.relation_embeddings, relations))

        for i in range(self.iteration):
            aggregator = self.aggregators[i]
            edge_vectors_next_iter = []
            for hop in range(self.iteration - i):
                vector = aggregator(self_vectors=edge_vectors[hop], neighbor_vectors=tf.reshape(
                    edge_vectors[hop + 1], [self.batch_size, -1, 2, self.n_neighbor, self.dim]))
                edge_vectors_next_iter.append(vector)
            edge_vectors = edge_vectors_next_iter

        res = tf.reshape(edge_vectors[0], [self.batch_size, self.dim])
        return res

    def _build_train(self):
        self.base_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.relations,
                                                                                       logits=self.scores))
        self.l2_loss = tf.nn.l2_loss(self.emb_mapping.weights)  # l2 loss of the embedding layer
        for aggregator in self.aggregators:
            self.l2_loss += self.l2_weight * tf.nn.l2_loss(aggregator.weights)  # l2 loss of each aggregator
        self.loss = self.base_loss + self.l2_loss

        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)

    def eval(self, sess, feed_dict):
        return sess.run(self.scores_normalized, feed_dict)
