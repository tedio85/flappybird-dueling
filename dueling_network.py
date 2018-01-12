import math
import numpy as np
import tensorflow as tf


class DuelingNetwork(object):
    def __init__(self, sess, hps, online=True, training=True):
        self.sess = sess
        self.hps = hps
        self.online = online
        self.training = training
        self.global_step = tf.train.get_or_create_global_step()


        scope_name = 'online' if online else 'target'
        with tf.variable_scope(scope_name):
            self._build_network()
            if online:
                self._build_train_op()


    def _build_network(self):
        # input
        input = tf.placeholder(
                    tf.float32,
                    shape=[None, self.hps.num_frames, self.hps.pic_height, self.hps.pic_width],
                    name='input_state')
        self.input_state = input

        # (batch_size, width, height, num_frame_stacked) = (None, 512, 288, 4)
        nhwc = tf.transpose(self.input_state, [0, 2, 3, 1])
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
                        rate=self.hps.dropout_rate,
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
                        rate=self.hps.dropout_rate,
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
                        rate=self.hps.dropout_rate,
                        training=self.training,
                        name='dropout3' )

        # flatten before split into value and advantage stream
        flat = tf.contrib.layers.flatten(bn3)

        # value stream, outputs 1 logit, as the value
        vs = tf.layers.dense(inputs=flat, units=512, activation=tf.nn.relu, name='value_stream_in')
        value = tf.layers.dense(inputs=vs, units=1, activation=None, name='value_stream_out')

        # action advantage stream, outputs 2 logits, one for each action
        advs = tf.layers.dense(inputs=flat, units=512, activation=tf.nn.relu, name='advantage_stream_in')
        advantage = tf.layers.dense(inputs=advs, units=self.hps.num_action, activation=None, name='advantage_stream_out') #(batch_size, 2)

        # merge the value stream and advantage stream
        mean_adv = tf.reduce_mean(advantage, axis=1, keep_dims=True, name='mean_adv') #(batch_size, 1)
        q_value = value + (advantage - mean_adv) #(batch_size, 2) - (batch_size, 1) using broadcast
        self.q_value = q_value
        self.pred = tf.argmax(q_value, axis=1) # select action with highest Q value

    def _build_train_op(self):

        target_q = tf.placeholder(tf.float32, shape=[None, 1], name='y_i') # y_i

        # TODO: need to compute Q(s,a|theta)

        loss = tf.square(target_q - estimated_q)

        # the mean and variance of batch norm layers has to be updated manually
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.hps.lr)
            grads_and_vars = optimizer.compute_gradients(loss)

            ######## for testing ########
            for i in grads_and_vars:
                print('(', i[0],',',i[1],')')
            #############################

            # since network has 2 streams, scale the gradients by 1/sqrt(2) as stated in the paper
            scale_factor = 1/math.sqrt(2)
            scaled_grads_and_vars = [(val*scale_factor, name) for val, name in scaled_grads_and_vars]

            # gradient clipping at 10 stated in paper
            clip = self.hps.clip_value
            clipped_grads_and_vars = [(tf.clip_by_value(val, -clip, clip), name) \
                                      for val, name in scaled_grads_and_vars]

            train_op = optimizer.apply_gradients(clipped_grads_and_vars, global_step=self.global_step)


    def _activation_summary(self, tensor):
        tensor_name = tensor.op.name
        tf.summary.histogram(tensor_name + '/activations', tensor)
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(tensor))

    def select_action(self, input_state):
        # epsilon-greedy
        if np.random.rand() < self.exploring_rate:
            action = np.random.choice(num_action)  # Select a random action
        else:
            input_state = np.asarray(input_state).transpose([1, 2, 0]) #(4, 512, 288) => (512, 288, 4)
            feed = {
                self.input_state: input_state[None, :]
            }
            action = self.sess.run(self.pred, feed_dict=feed)

        return action

    #def train(self, s1, a, r, t, s2):
