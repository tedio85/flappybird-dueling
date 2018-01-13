import os
os.environ["SDL_VIDEODRIVER"] = "dummy"  # make window not appear
import tensorflow as tf
import numpy as np
import skimage.color
import skimage.transform
from ple.games.flappybird import FlappyBird
from ple import PLE

import tensorflow as tf
import numpy as np

from replay_buffer import ReplayBuffer
from dueling_network import DuelingNetwork


def get_default_hparams(num_action=2, buffer_size=10**6):
    hparams = tf.contrib.training.HParams(
        buffer_size=buffer_size,
        num_action=num_action,
        pic_width=288,
        pic_height=512,
        num_frames=4,
        clip_value=10,
        exploring_rate=0.1,
        min_exploring_rate=1e-3,
        dropout_rate=0.1,  # droput_keep_prob = 1 - dropout_rate
        discount_factor=0.99,
        tau=0.001,
        lr=1e-4,
        batch_size=256,
        training_episodes=10**6)
    return hparams


if __name__ == '__main__':
    game = FlappyBird()
    # environment interface to game
    env = PLE(game, fps=30, display_screen=False)
    env.reset_game()
    num_action = len(env.getActionSet())

    hps = get_default_hparams(num_action)

    with tf.Session() as sess:
        # initialize network and buffer
        online = DuelingNetwork(sess, hps, online=True,
                                online_network=None, training=True)
        target = DuelingNetwork(sess, hps, online=False,
                                online_network=online, training=True)
        buffer = ReplayBuffer(hps.buffer_size)
        sess.run(tf.global_variables_initializer())

        # test if update_target_network() is functioning
        target.update_target_network()

        # populate replay buffer
        # while buffer.size() < self.hps.buffer_size:
