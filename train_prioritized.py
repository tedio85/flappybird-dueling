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
from max_heap import MaxHeap
from prioritized_buffer import Experience as ReplayBuffer
from dueling_prioritized import DuelingPrioritized

# constants
SAVE_VIDEO_AFTER_EPISODES = 500
SAVE_CHECKPOINT_AFTER_EPISODES = 1000

def write_log(log_dir, log_msg):
    with open(log_dir, 'a') as f:
        f.write(log_msg)


def get_default_hparams(num_action=2, buffer_size=10**5):
    hparams = tf.contrib.training.HParams(
        buffer_size=buffer_size,
        num_action=num_action,
        pic_width=72,
        pic_height=128,
        num_frames=4,
        clip_value=10,
        exploring_rate=0.1,
        min_exploring_rate=1e-3,
        dropout_rate=0.1,  # droput_keep_prob = 1 - dropout_rate
        discount_factor=0.99,
        tau=0.001,
        lr=1e-4,
        alpha=0.7, # alpha for replay buffer
        beta=0.5, # beta for replay buffer
        priority_epsilon=0.00001, # epsilon for calculating priority
        batch_size=1024,
        replay_period=128,
        training_episodes=300000,
        ckpt_path='/tmp/md/ted_tmp/flappybird/checkpoint_prioritized/',
        summary_path='/tmp/md/ted_tmp/flappybird/summary_prioritized/',
        anim_path='/tmp/md/ted_tmp/flappybird/anim_prioritized/',
        log_path='/tmp/md/ted_tmp/flappybird/train_log_prioritized.txt')
    return hparams


if __name__ == '__main__':
    game = FlappyBird()
    # environment interface to game
    env = PLE(game, fps=30, rng=np.random.RandomState(np.random.randint(low=0, high=20180114)), display_screen=False)
    env.reset_game()
    num_action = len(env.getActionSet())

    hps = get_default_hparams(num_action)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # initialize network and buffer
        online = DuelingPrioritized(sess, hps, online=True,
                                    online_network=None, training=True)
        target = DuelingPrioritized(sess, hps, online=False,
                                    online_network=online, training=True)
        maxheap = MaxHeap()
        buffer_conf = {
            'size': hps.buffer_size,
            'replace_old': True,
            'alpha': hps.alpha
        }
        buffer = ReplayBuffer(conf=buffer_conf)
        saver = tf.train.Saver(max_to_keep=15)
        writer = tf.summary.FileWriter(hps.summary_path, sess.graph)
        sess.run(tf.global_variables_initializer())

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


        # copy online network
        target.copy_online_network()

        # populate buffer
        input_screens = [online.preprocess(env.getScreenGrayscale())]*4
        records = 0
        while records < hps.buffer_size+100:
            game = FlappyBird()
            env = PLE(
                game,
                fps=30,
                rng=np.random.RandomState(np.random.randint(low=0, high=2**32-1)),
                display_screen=False)
            env.reset_game()
            print('current buffer size: {}/{}'.format(records, hps.buffer_size))
            write_log(hps.log_path,
                      'current buffer size: {}/{}\n'.format(records, hps.buffer_size))

            while not env.game_over():
                a = target.select_action(input_screens[-4:])
                r = env.act(env.getActionSet()[a])
                te = env.game_over()
                input_screens.append(online.preprocess(env.getScreenGrayscale()))

                # add (abs(priority)+e, s, a, r, t, s2) to max heap
                td = online.get_TD_error(input_screens[-5:-1], a, r, te, input_screens[-4:],\
                                               target)
                priority = np.absolute(td)
                data = (priority, input_screens[-5:-1], a, r, input_screens[-4:], te)
                maxheap.push(data)

                # add the sample with maximal abs(TD_error) to replay buffer
                sample, priority = maxheap.top()
                buffer.store(sample)
                records += 1

        buffer.rebalance()
        print('buffer full!')
        write_log(hps.log_path, 'buffer full!\n')


        # begin trial and update
        episode = init_episode
        update_global_step = tf.assign(online.global_step, episode)
        for episode in range(init_episode, hps.training_episodes):
            sess.run(update_global_step)

            # reset game
            game = FlappyBird()
            env = PLE(
                game,
                fps=30,
                rng=np.random.RandomState(np.random.randint(low=0, high=20180114)),
                display_screen=False)
            env.reset_game()

            # for model input
            input_screens = [online.preprocess(env.getScreenGrayscale())] * 4

            # for video clip
            video_frames = [env.getScreenRGB()]
            if episode % SAVE_VIDEO_AFTER_EPISODES == 0:
                online.shutdown_explore()

            step = 0
            cum_reward = 0
            maxheap.clear()
            while not env.game_over():
                a = online.select_action(input_screens[-4:])
                r = env.act(env.getActionSet()[a])
                te = env.game_over()
                input_screens.append(online.preprocess(env.getScreenGrayscale()))

                # add (abs(priority), s, a, r, t, s2) to max heap
                td = online.get_TD_error(input_screens[-5:-1], a, r, te, input_screens[-4:],\
                                               target)
                priority = np.absolute(td)
                data = (priority, input_screens[-5:-1], a, r, input_screens[-4:], te)
                maxheap.push(data)

                # add the sample with maximal abs(TD_error) to replay buffer
                sample, priority = maxheap.top()
                buffer.store(sample)

                cum_reward += r
                step += 1

                 # record frame
                if episode % SAVE_VIDEO_AFTER_EPISODES == 0:
                    video_frames.append(env.getScreenRGB())


                # sample batch and update online & target network
                if (step-1) % hps.replay_period == 0:
                    experience, weights, indices = buffer.sample(episode, hps.batch_size)
                    s, a, r, s2, t = utils.list_to_batches(experience)
                    TD_error = online.get_TD_error(s, a, r, t, s2, target)
                    TD_list = utils.flatten_array_as_list(TD_error)

                    # update transition priority
                    buffer.update_priority(indices, TD_list)

                    # train the online network
                    loss, summary = online.train(s, a, r, t, s2, weights, target)
                    writer.add_summary(summary, global_step=episode)
                    target.update_target_network()
                    buffer.rebalance()


            # update exploring rate
            online.update_exploring_rate(episode)
            target.update_exploring_rate(episode)

            # log information and summaries
            log_info = \
            "[{}] time live:{} cumulated reward: {} exploring rate: {:.4f} loss: {:.4f}".format(
                                            episode, step, cum_reward, online.exp_rate, loss)
            print(log_info)
            write_log(hps.log_path, log_info+'\n')


            # save checkpoint
            if episode % SAVE_CHECKPOINT_AFTER_EPISODES == 0:
                saver.save(sess, os.path.join(hps.ckpt_path, 'prioritized.ckpt-{}'.format(episode)))

            # save video clip
            if episode % SAVE_VIDEO_AFTER_EPISODES == 0:
                video_frames = [np.transpose(frame, axes=(1,0,2)) for frame in video_frames]
                utils.make_anim(video_frames, episode, anim_dir=hps.anim_path, true_image=True)
