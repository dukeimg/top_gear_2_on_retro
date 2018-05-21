#!/usr/bin/env python

import argparse
import retro
import numpy as np
import tensorflow as tf

from collections import deque
from gym.envs.classic_control.rendering import SimpleImageViewer

from dqn.model import DQNetwork

parser = argparse.ArgumentParser()
parser.add_argument('model', nargs='?', help='model to use', default='latest_model')
parser.add_argument('state', nargs='?', help='the initial state file to load, minus the extension')
parser.add_argument('--verbose', '-v', action='count', default=1, help='increase verbosity (can be specified multiple times)')
parser.add_argument('--quiet', '-q', action='count', default=0, help='decrease verbosity (can be specified multiple times)')
args = parser.parse_args()
verbosity = args.verbose - args.quiet

viewer = SimpleImageViewer()


def stack_frames(stacked_frames, state):
    # Append frame to deque, automatically removes the oldest frame
    stacked_frames.append(state)

    # Build the stacked state (first dimension specifies different frames)
    stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_state


def render(img):
    viewer.imshow(img[10:208, 0:256])


def process_image(img):
    img = img[135:208, 42:214]
    return np.mean(img, -1)


env = retro.make('TopGear2-Genesis', args.state or retro.STATE_DEFAULT)

action_space = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
]


# Model hyperparameters
stack_size = 4
frame_shape = [73, 172]
stacked_frame_shape = [*frame_shape, stack_size]
state_size = stacked_frame_shape      # Our input is a stack of 4 frames hence 84x84x4 (Width, height, channels)
action_size = action_space.__len__()
learning_rate = 0.0002      # Alpha (aka learning rate)

stacked_frames = deque([np.zeros(frame_shape, dtype=np.int) for i in range(stack_size)], maxlen=4)
DQNetwork = DQNetwork(state_size, action_size, learning_rate)

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "./models/{}.ckpt".format(args.model or 'latest_model'))
    done = False

    env.reset()

    step = 0
    while not done:
        step += 1
        render(env.img)
        frame = process_image(env.img)
        state = stack_frames(stacked_frames, frame)
        # Take the biggest Q value (= the best action)
        Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})
        action = np.argmax(Qs)
        action = action_space[int(action)]
        next_state, reward, done, info = env.step(action)
    print('Race finished at t=%i, position: %i' % (step, info.get('position')))
    exit(0)
