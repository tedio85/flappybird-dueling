import moviepy.editor as mpy
import os

def make_anim(images, episode, anim_dir='', fps=60, true_image=False):
    duration = len(images) / fps

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
