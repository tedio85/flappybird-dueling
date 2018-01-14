import math
import numpy as np
import tensorflow as tf


class DuelingNetwork(object):
    def __init__(self, sess, hps, online=True, online_network=None, training=True):
        self.sess = sess
        self.hps = hps
        self.online = online
        self.online_network = online_network # paramter for target network
        self.training = training
        self.global_step = tf.train.get_or_create_global_step()
        self.exp_rate = self.hps.exploring_rate

        scope_name = 'online' if online else 'target'
        self.scope_name = scope_name
        with tf.variable_scope(scope_name):
            self._build_network()
            self.params_for_update = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.scope_name)
            if online:
                self._build_train_op()
                self._build_summary()
            else:
                self._build_update_op(self.online_network)


    def _build_network(self):
        # inputs: state & action
        # shape(None, 4, 512, 288)
        input_state = tf.placeholder(
                    tf.float32,
                    shape=[None, self.hps.num_frames, self.hps.pic_height, self.hps.pic_width],
                    name='input_state')
        action = tf.placeholder(tf.int32, shape=[None], name='action')
        self.input_state = input_state
        self.action = action


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
        q_value = value + (advantage - mean_adv) #(batch_size, 2) - (batch_size, 1) using broadcast becomes (batch_size, 2)
        self.q_value = q_value # Q(s,a,theta) for all a, shape (batch_size, num_action)


        # get the action with highest Q, or argmax_a' Q(s',a'|theta)
        self.argmax_a = tf.argmax(q_value, axis=1)


        # Q(s, a) for selected action, shape (batch_size, 1)
        # (0, a0), (1, a1), (2, a2) ....
        index = tf.stack([tf.range(tf.shape(self.action)[0]), self.action], axis=1)
        # get the position a0 of q_value[0], a1 of q_value[1] ....
        self.estimatedQ = tf.gather_nd(q_value, index)


    def _build_train_op(self):
        """Used for training the online network"""
        targetQ = tf.placeholder(tf.float32, shape=[None, 1], name='y_i') # y_i
        self.targetQ = targetQ

        # (y_i - Q(s,a|theta))^2
        loss = tf.square(targetQ - self.estimatedQ)
        self.loss = loss

        # the mean and variance of batch norm layers has to be updated manually
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.hps.lr)
            grads_and_vars = optimizer.compute_gradients(loss)

            ######## for testing ########
            #for i in grads_and_vars:
                #print('(', i[0],',',i[1],')')
            #############################

            # since network has 2 streams, scale the gradients by 1/sqrt(2) as stated in the paper
            scale_factor = 1/math.sqrt(2)
            scaled_grads_and_vars = [(val*scale_factor, name) for val, name in grads_and_vars]

            # gradient clipping at 10 stated in paper
            clip = self.hps.clip_value
            clipped_grads_and_vars = [(tf.clip_by_value(val, -clip, clip), name) \
                                      for val, name in scaled_grads_and_vars]

            self.train_op = optimizer.apply_gradients(clipped_grads_and_vars, global_step=self.global_step)
            self._gradient_summary(clipped_grads_and_vars)

    def _build_update_op(self, online_network):
        """Used for updating the target network"""
        online_params = online_network.get_params_for_update()
        target_params = self.get_params_for_update()
        num_params = len(target_params)
        assert num_params == len(online_params)

        update_op = [
            target_params[i].assign(
                target_params[i] * self.hps.tau + \
                online_params[i] * (1 - self.hps.tau)
            )
            for i in range(num_params)
        ]
        self.update_op = update_op


    def _activation_summary(self, tensor):
        tensor_name = tensor.op.name
        tf.summary.histogram(tensor_name + '/activations', tensor)
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(tensor))

    def _gradient_summary(self, grads_and_vars):
        for grad, var in grads_and_vars:
            tf.summary.histogram(var.name + '/gradient', grad)

    def _build_summary(self):
        tf.summary.scalar('exploring_rate', self.hps.exploring_rate)
        self.summary = tf.summary.merge_all()


    def select_action(self, input_state, sess=None):
        # epsilon-greedy
        if np.random.rand() < self.exp_rate:
            action = np.random.choice(self.hps.num_action)  # Select a random action
        else:
            input_state = np.asarray(input_state)
            feed = {
                self.input_state: input_state[None, :]
            }
            act = self.sess.run(self.argmax_a, feed_dict=feed)
            action = act[0]

        return action

    def train(self, s, a, r, t, s2, target_network):
        """Used for online network training"""
        if not online:
            raise Exception('train() is for the online network, not the target network')

        # get argmax_a' Q(s',a' |theta)
        feed_online = { self.input_state: s2 }
        a_max = sess.run(self.argmax_a, feed_dict=feed_online)

        # get y_i
        targetQ = target_network.get_targetQ(s2, a_max, r, t)

        # compute loss in _build_train_op, loss = (y_i - Q(s,a|theta))^2
        feed_online_yi = {
            self.targetQ: targetQ,
            self.input_state: s,
            self.action: a
        }
        loss, summary, _ = self.sess.run(
                            [self.loss, self.summary, self.train_op],
                            feed_dict=feed_online_yi)

        return loss, summary

    def get_targetQ(self, s2, a_max, r, t):
        """Get y_i from target network"""
        if online:
            raise Exception('get_targetQ() is for target network, not the online network!')

        feed = {
            self.input_state: s2,
            self.action: a_max
        }

        # Q(s', a_max(s'|theta)   | theta')
        Q = self.sess.run(self.estimatedQ, feed_dict=feed) # shape (batch_size, 1)

        # tensor for non-terminal states
        non_term = r + self.hps.discount_factor * Q
        cond = tf.equal(t, True)

        # result[i] = r if terminal==True
        # result[i] = r + discount_factor * Q(s', a_max(s'|theta)   | theta')
        result = tf.where(cond, r, non_term)

        return result

    def get_params_for_update(self):
        return self.params_for_update

    def update_target_network(self):
        """Used for target network to update itself"""
        if self.online:
            raise Exception('update_target_network() is for updating target network,\
                             not the online network!')

        self.sess.run(self.update_op)

    def update_exploring_rate(self, episode):
        if self.hps.exploring_rate > self.hps.min_exploring_rate:
            self.exp_rate -= (self.hps.exploring_rate - self.hps.min_exploring_rate) / 3000000

    def shutdown_explore(self):
        # make action selection greedy
        self.exp_rate = 0


    def preprocess(self, screen):
        screen = skimage.transform.resize(screen, [144, 256])
        return np.transpose(screen, [1, 0])
