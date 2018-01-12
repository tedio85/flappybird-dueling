import os
os.environ["SDL_VIDEODRIVER"] = "dummy" # make window not appear
import tensorflow as tf
import numpy as np
import skimage.color
import skimage.transform
from ple.games.flappybird import FlappyBird
from ple import PLE

import tensorflow as tf
import numpy as np

from dueling_network import DuelingNetwork

def get_default_hparams(num_action=2):
    hparams = tf.contrib.training.HParams(
                  num_action=num_action,
                  pic_width=288,
                  pic_height=512,
                  num_frames=4,
                  clip_value=10,
                  dropout_rate=0.1,
                  lr=1e-4,
                  batch_size=64,
                  training_epochs=10000)
    return hparams

if __name__ == '__main__':
    game = FlappyBird()
    env = PLE(game, fps=30, display_screen=False) # environment interface to game
    env.reset_game()
    num_action = len(env.getActionSet())

    hps = get_default_hparams(num_action)

    with tf.Session() as sess:
        dnet = DuelingNetwork(sess, hps, training=True)
