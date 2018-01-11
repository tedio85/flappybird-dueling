import numpy as np
import tensorflow as tf

class DuelingNetwork(object):
    def __init__(self, hps, training=True):
        self.hps = hps
        self.training = training

    def _build_network(self):
        # input
        # (batch_size, width, height, num_frame_stacked)
        nhwc = tf.transpose(inputs.frames, [0, 2, 3, 1])
        self._activation_summary(nhwc)

        # conv1
        conv1 = tf.layers.conv2d(
            nhwc,
            filters=32,
            kernel_size=[8, 8],
            strides=[4, 4],
            activation=tf.nn.relu,
            name='conv1')
        self._activation_summary(conv1)
        bn1 = tf.layers.batch_normalization(
                        conv1,
                        axis=-1,
                        training=self.training,
                        name='bn1')
        dp1 = tf.layers.dropout(
                        bn1,
                        rate=self.hps.droput_rate,
                        training=self.training,
                        name='dropout1' )

        # conv2
        conv2 = tf.layers.conv2d(
            dp1,
            filters=64,
            kernel_size=[4, 4],
            strides=[2, 2],
            activation=tf.nn.relu,
            name='conv2')
        self._activation_summary(conv2)
        bn2 = tf.layers.batch_normalization(
                        conv2,
                        axis=-1,
                        training=self.training,
                        name='bn2')
        dp2 = tf.layers.dropout(
                        bn2,
                        rate=self.hps.droput_rate,
                        training=self.training,
                        name='dropout2' )

        # conv3
        conv3 = tf.layers.conv2d(
            dp2,
            filters=64,
            kernel_size=[3, 3],
            strides=[1, 1],
            activation=tf.nn.relu,
            name='conv3')
        self._activation_summary(conv3)
        bn3 = tf.layers.batch_normalization(
                        conv3,
                        axis=-1,
                        training=self.training,
                        name='bn3')

        dp3 = tf.layers.dropout(
                        bn3,
                        rate=self.hps.droput_rate,
                        training=self.training,
                        name='dropout3' )

        # flatten before split into value and advantage stream
        flat = tf.contrib.layers.flatten(bn3)

        # value stream, outputs 1 logit, as the value
        vs = tf.layers.dense(inputs=flat, units=512, activation=tf.nn.relu, name='value_stream_in')
        value = tf.layers.dense(inputs=vs, units=1, activation=None, name='value_stream_out')

        # action advantage stream, outputs 2 logits, one for each action
        advs = tf.layers.dense(inputs=flat, units=512, activation=tf.nn.relu, name='advantage_stream_in')
        advantage = tf.layers.dense(inputs=advs, units=2, activation=None, name='advantage_stream_out')

        # merge the value stream and advantage stream
        mean_adv = tf.reduce_mean(advantage, axis=1, keep_dims=True, name='mean_adv')
        q_value = value + (advantage - mean_adv)
        self.q_value = q_value

    def _build_train_op(self):


    def _activation_summary(self, tensor):
        tensor_name = tensor.op.name
        tf.summary.histogram(tensor_name + '/activations', tensor)
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(tensor))
