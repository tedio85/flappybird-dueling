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
        training_episodes=10**6,
        ckpt_path='/tmp/md/ted_tmp/flappybird/',
        summary_path='/tmp/md/ted_tmp/flappybird/summary_ddqn')
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

        # populate buffer
        input_screens = [online.preprocess(env.getScreenGrayscale())]*4
        while buffer.size() < hps.batch_size*10:
            game = FlappyBird()
            env = PLE(
                game,
                fps=30,
                rng=np.random.RandomState(np.random.randint(low=0, high=200000)),
                display_screen=False)
            env.reset_game()
            print('current buffer size: {}'.format(buffer.size()))
            while not env.game_over():
                a = online.select_action(input_screens[-4:])
                r = env.act(env.getActionSet()[a])
                te = env.game_over()
                input_screens.append(online.preprocess(env.getScreenGrayscale()))
                buffer.add(input_screens[-5:-1], a, r, te, input_screens[-4:])
        print('buffer full!')

        # restore previously stored
        ckpt = tf.train.get_checkpoint_state(self.hps.ckpt_dir)
        init_episode = 0
        if ckpt and ckpt.model_checkpoint_path:
            # if checkpoint exists
            print('restore from ', ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            # assume the name of checkpoint is like '.../ddqn-1000'
            init_episode = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            sess.run(tf.assign(online.global_step, init_episode))

        for epsode in range(init_episode, self.hps.training_episodes):
            sess.run(tf.assign(online.global_step, episode))

            # reset game
            game = FlappyBird()
            env = PLE(
                game,
                fps=30,
                rng=np.random.RandomState(np.random.randint(low=0, high=200000)),
                display_screen=False)
            env.reset_game()

            
