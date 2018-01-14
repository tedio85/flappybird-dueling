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
import moviepy.editor as mpy

import utils
from replay_buffer import ReplayBuffer
from dueling_network import DuelingNetwork

# constants
SAVE_VIDEO_AFTER_EPISODES = 500
SAVE_CHECKPOINT_AFTER_EPISODES = 1000

def write_log(log_dir, log_msg):
    with open(log_dir, 'a') as f:
        f.write(log_msg)
    

def get_default_hparams(num_action=2, buffer_size=10**6):
    hparams = tf.contrib.training.HParams(
        buffer_size=buffer_size,
        num_action=num_action,
        pic_width=144,
        pic_height=256,
        num_frames=4,
        clip_value=10,
        exploring_rate=0.1,
        min_exploring_rate=1e-3,
        dropout_rate=0.1,  # droput_keep_prob = 1 - dropout_rate
        discount_factor=0.99,
        tau=0.001,
        lr=1e-4,
        batch_size=64,
        training_episodes=10**6,
        ckpt_path='/tmp/md/ted_tmp/flappybird/checkpoint_ddqn/',
        summary_path='/tmp/md/ted_tmp/flappybird/summary_ddqn/',
        anim_path='/tmp/md/ted_tmp/flappybird/anim_ddqn/',
        log_path='/tmp/md/ted_tmp/flappybird/train_log.txt')
    return hparams


if __name__ == '__main__':
    game = FlappyBird()
    # environment interface to game
    env = PLE(game, fps=30, display_screen=False)
    env.reset_game()
    num_action = len(env.getActionSet())

    hps = get_default_hparams(num_action)

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    with tf.Session(config=config) as sess:
        # initialize network and buffer
        online = DuelingNetwork(sess, hps, online=True,
                                online_network=None, training=True)
        target = DuelingNetwork(sess, hps, online=False,
                                online_network=online, training=True)
        buffer = ReplayBuffer(hps.buffer_size)
        saver = tf.train.Saver(max_to_keep=15)
        writer = tf.summary.FileWriter(hps.summary_path, sess.graph)
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
            print('current buffer size: {}/{}'.format(buffer.size(), hps.batch_size*10))
            write_log(hps.log_path, 
                      'current buffer size: {}/{}\n'.format(buffer.size(), hps.batch_size*10))
            
            while not env.game_over():
                a = target.select_action(input_screens[-4:])
                r = env.act(env.getActionSet()[a])
                te = env.game_over()
                input_screens.append(online.preprocess(env.getScreenGrayscale()))
                buffer.add(input_screens[-5:-1], a, r, te, input_screens[-4:])
        print('buffer full!')
        write_log(hps.log_path, 'buffer full!\n')

        # restore previously stored checkpoint
        ckpt = tf.train.get_checkpoint_state(hps.ckpt_path)
        init_episode = 0
        if ckpt and ckpt.model_checkpoint_path:
            # if checkpoint exists
            print('restore from ', ckpt.model_checkpoint_path)
            write_log(hps.log_path, 'restore from '+ckpt.model_checkpoint_path+'\n')
            saver.restore(sess, ckpt.model_checkpoint_path)
            # assume the name of checkpoint is like '.../ddqn-1000'
            init_episode = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            sess.run(tf.assign(online.global_step, init_episode))


        for episode in range(init_episode, hps.training_episodes):
            sess.run(tf.assign(online.global_step, episode))

            # reset game
            game = FlappyBird()
            env = PLE(
                game,
                fps=30,
                rng=np.random.RandomState(np.random.randint(low=0, high=200000)),
                display_screen=False)
            env.reset_game()

            # for model input
            input_screens = [online.preprocess(env.getScreenGrayscale())] * 4

            # for video clip
            video_frames = [env.getScreenRGB()]
            if episode % SAVE_VIDEO_AFTER_EPISODES == 0:
                online.shutdown_explore()

            step = 0
            while not env.game_over():
                cum_reward = 0
                a = online.select_action(input_screens[-4:])
                r = env.act(env.getActionSet()[a])
                te = env.game_over()
                input_screens.append(online.preprocess(env.getScreenGrayscale()))
                buffer.add(input_screens[-5:-1], a, r, te, input_screens[-4:])
                cum_reward += r
                step += 1

                 # record frame
                if episode % SAVE_VIDEO_AFTER_EPISODES == 0:
                    video_frames.append(env.getScreenRGB())



            # sample batch and update online & target network
            s, a, r, t, s2 = buffer.sample_batch(hps.batch_size)
            loss, summary = online.train(s, a, r, t, s2, target)
            target.update_target_network()


            # update exploring rate
            online.update_exploring_rate(episode)
            target.update_exploring_rate(episode)

            # log information and summaries
            log_info = \
            "[{}] time live:{} cumulated reward: {} exploring rate: {:.4f} loss: {:.4f}".format(
                                        episode, step, cum_reward, online.exp_rate, loss)
            print(log_info)
            write_log(hps.log_path, log_info+'\n')
            writer.add_summary(summary, global_step=episode)

            # save checkpoint
            if episode % SAVE_CHECKPOINT_AFTER_EPISODES == 0:
                saver.save(sess, os.path.join(hps.ckpt_path, 'ddqn.ckpt-{}'.format(episode)))

            # save video clip
            if episode % SAVE_VIDEO_AFTER_EPISODES == 0:
                utils.make_anim(video_frames, episode, anim_dir=hps.anim_path)
