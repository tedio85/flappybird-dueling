#import moviepy.editor as mpy
import os
import numpy as np
import itertools

def list_to_batches(sample_batch):
    array = np.asarray(sample_batch)
    s = array[:,0].reshape(-1,1)
    a = array[:,1].reshape(-1,1)
    r = array[:,2].reshape(-1,1)
    t = array[:,3].reshape(-1,1)
    s2 = array[:,4].reshape(-1,1)

    return s, a, r, t, s2

def flatten_array_as_list(batch):
    """
        For an array like
        [[1, 1],
         [2, 2],
         [3, 3],
         [4, 4],
         [5, 5]]

         returns [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    """
    nested_list = np.ndarray.tolist(batch)
    return list(itertools.chain.from_iterable(nested_list))


def make_anim(images, episode, anim_dir='', fps=60, true_image=False):
    duration = len(images) / fps
    import moviepy.editor as mpy

    def make_frame(t):
        try:
            x = images[int(len(images) / duration * t)]
        except:
            x = images[-1]

        if true_image:
            return x.astype(np.uint8)
        else:
            return ((x + 1) / 2 * 255).astype(np.uint8)

    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.fps = fps
    save_path = os.path.join(anim_dir, 'DDQN-{}.mp4'.format(episode))
    clip.write_videofile(save_path)
