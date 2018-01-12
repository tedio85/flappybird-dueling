import tensorflow as tf
import numpy as np

from dueling_network import DuelingNetwork

def get_default_hparams():
    hparams = tf.contrib.training.HParams(
                  pic_width=288,
                  pic_height=512,
                  num_frames=4,
                  dropout_rate=0.1,
                  lr=1e-4,
                  batch_size=64,
                  training_epochs=10000)
    return hparams

if __name__ == '__main__':
    hps = get_default_hparams()
    dnet = DuelingNetwork(hps, training=True)
