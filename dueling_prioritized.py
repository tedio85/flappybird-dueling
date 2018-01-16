import math
import numpy as np
import tensorflow as tf

from dueling_network import DuelingNetwork

class DuelingPrioritized(DuelingNetwork):
    def __init__(self, sess, hps, online=True, online_network=None, training=True):
        super().__init__(sess, hps, online, online_network, training)

    def _build_train_op(self):
        targetQ = tf.placeholder(tf.float32, shape=[None, 1], name='y_i') # y_i
        weights = tf.placeholder(tf.float32, shape=[None, 1], name='priority_weights')
        self.targetQ = targetQ
        self.weights = weights

        # get TD_error
        TD_error = targetQ - self.estimatedQ
        self.TD_error = TD_error # shape (batch_size, 1)


        # accumulate weight change(delta)
        loss = tf.reduce_sum(weights * TD_error * self.estimatedQ)
        self.loss = loss


        # the mean and variance of batch norm layers has to be updated manually
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=self.hps.lr)
            grads_and_vars = optimizer.compute_gradients(loss)

            # since network has 2 streams, scale the gradients by 1/sqrt(2) as stated in the paper
            scale_factor = 1/math.sqrt(2)
            scaled_grads_and_vars = [(val*scale_factor, name) for val, name in grads_and_vars]

            # gradient clipping at 10 stated in paper
            clip = self.hps.clip_value
            clipped_grads_and_vars = [(tf.clip_by_value(val, -clip, clip), name) \
                                      for val, name in scaled_grads_and_vars]

            self.train_op = optimizer.apply_gradients(clipped_grads_and_vars, global_step=self.global_step)
            self._gradient_summary(clipped_grads_and_vars)

    def get_TD_error(self, s, a, r, t, s2, target_network):
        if not self.online:
            raise Exception('get_TD_error() is for the online network, not the target network')
        
        s = np.asarray(s)
        s2 = np.asarray(s2)

        if len(s.shape) < 4:
            s = s[None, :]
        if len(s2.shape) < 4:
            s2 = s2[None, :]
            
        
        # get argmax_a' Q(s',a' |theta)
        feed_online = { self.input_state: s2 }
        a_max = self.sess.run(self.argmax_a, feed_dict=feed_online)

        # get y_i
        if np.isscalar(a):
            a = np.asarray(a).reshape(-1,)
            r = np.asarray(r).reshape(-1,)
            t = np.asarray(t).reshape(-1,)
        targetQ = target_network.get_targetQ(s2, a_max, r, t)

        feed_online_yi = {
            self.targetQ: targetQ,
            self.input_state: s,
            self.action: a
        }
        # shape (1, 1) for single sample, (batch_size, 1) for a batch samples
        TD_error = self.sess.run(self.TD_error, feed_dict=feed_online_yi)

        if TD_error.shape[0] == 1:
            return TD_error[0][0]
        else:
            return TD_error

    def train(self, s, a, r, t, s2, weights, target_network):
        """Used for online network training"""
        if not self.online:
            raise Exception('train() is for the online network, not the target network')
            
        s = np.asarray(s)
        s2 = np.asarray(s2)
        
        if len(s.shape) < 4:
            s = s[None, :]
        if len(s2.shape) < 4:
            s2 = s2[None, :]
            

        # get argmax_a' Q(s',a' |theta)
        feed_online = { self.input_state: s2 }
        a_max = self.sess.run(self.argmax_a, feed_dict=feed_online)

        # get y_i
        targetQ = target_network.get_targetQ(s2, a_max, r, t)

        # compute loss in _build_train_op
        weights = np.asarray(weights).reshape(-1, 1)
        feed_online_yi = {
            self.targetQ: targetQ,
            self.input_state: s,
            self.action: a,
            self.weights: weights
        }
        loss, summary, _ = self.sess.run(
                            [self.loss, self.summary, self.train_op],
                            feed_dict=feed_online_yi)

        return loss, summary
