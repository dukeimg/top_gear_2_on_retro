#!/usr/bin/env python

import argparse
import retro
import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices
import random                # Handling random number generation
from collections import deque# Ordered collection with ends
import matplotlib.pyplot as plt # Display graphs
from gym.envs.classic_control.rendering import SimpleImageViewer

parser = argparse.ArgumentParser()
parser.add_argument('game', default='TopGear2-Genesis', help='the name or path for the game to run')
parser.add_argument('state', nargs='?', help='the initial state file to load, minus the extension')
parser.add_argument('--scenario', '-s', default='scenario', help='the scenario file to load, minus the extension')
parser.add_argument('--record', '-r', action='store_true', help='record bk2 movies')
parser.add_argument('--verbose', '-v', action='count', default=1, help='increase verbosity (can be specified multiple times)')
parser.add_argument('--quiet', '-q', action='count', default=0, help='decrease verbosity (can be specified multiple times)')
args = parser.parse_args()
verbosity = args.verbose - args.quiet

viewer = SimpleImageViewer()


def stack_frames(stacked_frames, state):
    # Preprocess frame
    frame = state.img

    # Append frame to deque, automatically removes the oldest frame
    stacked_frames.append(frame)

    # Build the stacked state (first dimension specifies different frames)
    stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_state


def render(img):
    viewer.imshow(img[10:208, 0:256])


env = retro.make(args.game, args.state or retro.STATE_DEFAULT, scenario=args.scenario, record=args.record)

# MODEL HYPERPARAMETERS
state_size = [198, 256, 3, 4]      # Our input is a stack of 4 frames hence 84x84x4 (Width, height, channels)
action_size = env.action_space.__sizeof__()
learning_rate = 0.0002      # Alpha (aka learning rate)

# TRAINING HYPERPARAMETERS
total_episodes = 5000        # Total episodes for training
max_steps = 100              # Max possible steps in an episode
batch_size = 64

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability
decay_rate = 0.0001            # exponential decay rate for exploration prob

# Q learning hyperparameters
gamma = 0.99

# MEMORY HYPERPARAMETERS
pretrain_length = batch_size
memory_size = 50000

# PREPROCESSING HYPERPARAMETERS
stack_size = 4

# MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True


stacked_frames = deque([np.zeros((198, 256, 3), dtype=np.int) for i in range(stack_size)], maxlen=4)

try:
    while True:
        ob = env.reset()
        t = 0
        totrew = 0
        while True:
            ac = env.action_space.sample()
            ob, rew, done, info = env.step([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            t += 1
            if t % 10 == 0:
                if verbosity > 1:
                    infostr = ''
                    if info:
                        infostr = ', info: ' + ', '.join(['%s=%i' % (k, v) for k, v in info.items()])
                    print(('t=%i' % t) + infostr)
                render(env.img)
            totrew += rew
            if verbosity > 0:
                if rew > 0:
                    print('t=%i got reward: %d, current reward: %d' % (t, rew, totrew))
                if rew < 0:
                    print('t=%i got penalty: %d, current reward: %d' % (t, rew, totrew))
            if done:
                render(env.img)
                try:
                    if verbosity >= 0:
                        print("done! total reward: time=%i, reward=%d" % (t, totrew))
                        input("press enter to continue")
                        print()
                    else:
                        input("")
                except EOFError:
                    exit(0)
                break
except KeyboardInterrupt:
    exit(0)
