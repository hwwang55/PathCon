import tensorflow as tf
from abc import abstractmethod

AGGREGATOR_ID = {}


def get_layer_id(aggregator_name=''):
    if aggregator_name not in AGGREGATOR_ID:
        AGGREGATOR_ID[aggregator_name] = 0
        return 0
    else:
        AGGREGATOR_ID[aggregator_name] += 1
        return AGGREGATOR_ID[aggregator_name]


class Aggregator(object):
    def __init__(self, batch_size, input_dim, output_dim, dropout, act, self_included, name):
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.act = act
        self.self_included = self_included
        self.name = name

    def __call__(self, self_vectors, neighbor_vectors):
        # neighbor_vectors: [batch_size, -1, 2, n_neighbor, input_dim]
        entity_vectors = tf.reduce_mean(neighbor_vectors, axis=-2)  # [batch_size, -1, 2, input_dim]
        outputs = self._call(self_vectors, entity_vectors)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, entity_vectors):
        # self_vectors: [batch_size, -1, input_dim]
        # entity_vectors: [batch_size, -1, 2, input_dim]
        pass


class MeanAggregator(Aggregator):
    def __init__(self, batch_size, input_dim, output_dim, dropout, act=lambda x: x, self_included=True, name=None):
        super(MeanAggregator, self).__init__(batch_size, input_dim, output_dim, dropout, act, self_included, name)

        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(shape=[self.input_dim, self.output_dim],
                                           initializer=tf.contrib.layers.xavier_initializer(),
                                           name='weights')
            self.bias = tf.get_variable(shape=[self.output_dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, entity_vectors):
        # self_vectors: [batch_size, -1, input_dim]
        # entity_vectors: [batch_size, -1, 2, input_dim]

        output = tf.reduce_mean(entity_vectors, axis=-2)  # [batch_size, -1, input_dim]
        if self.self_included:
            output += self_vectors
        output = tf.reshape(output, [-1, self.input_dim])  # [-1, input_dim]
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)  # [-1, input_dim]
        output = tf.matmul(output, self.weights) + self.bias  # [-1, output_dim]
        output = tf.reshape(output, [self.batch_size, -1, self.output_dim])  # [batch_size, -1, output_dim]

        return self.act(output)


class ConcatAggregator(Aggregator):
    def __init__(self, batch_size, input_dim, output_dim, dropout, act=lambda x: x, self_included=True, name=None):
        super(ConcatAggregator, self).__init__(batch_size, input_dim, output_dim, dropout, act, self_included, name)

        with tf.variable_scope(self.name):
            multiplier = 3 if self_included else 2
            self.weights = tf.get_variable(shape=[self.input_dim * multiplier, self.output_dim],
                                           initializer=tf.contrib.layers.xavier_initializer(),
                                           name='weights')
            self.bias = tf.get_variable(shape=[self.output_dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, entity_vectors):
        # self_vectors: [batch_size, -1, input_dim]
        # entity_vectors: [batch_size, -1, 2, input_dim]

        output = tf.reshape(entity_vectors, [-1, self.input_dim * 2])  # [-1, input_dim * 2]
        if self.self_included:
            self_vectors = tf.reshape(self_vectors, [-1, self.input_dim])  # [-1, input_dim]
            output = tf.concat([self_vectors, output], axis=-1)  # [-1, input_dim * 3]
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)  # [-1, input_dim * 2 or 3]
        output = tf.matmul(output, self.weights) + self.bias  # [-1, output_dim]
        output = tf.reshape(output, [self.batch_size, -1, self.output_dim])  # [batch_size, -1, output_dim]

        return self.act(output)
