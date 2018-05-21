#!/usr/bin/env python

import random
import datetime
import subprocess

import argparse
import retro
import numpy as np
import tensorflow as tf

from collections import deque
from gym.envs.classic_control.rendering import SimpleImageViewer

from dqn.memory import Memory
from dqn.model import DQNetwork

parser = argparse.ArgumentParser()
parser.add_argument('state', nargs='?', help='the initial state file to load, minus the extension')
parser.add_argument('scenario', nargs='?', default='scenario', help='the scenario file to load, minus the extension')
parser.add_argument('--verbose', '-v', action='count', default=1, help='increase verbosity (can be specified multiple times)')
parser.add_argument('--quiet', '-q', action='count', default=0, help='decrease verbosity (can be specified multiple times)')
args = parser.parse_args()
verbosity = args.verbose - args.quiet

action_space = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
]

possible_actions = np.identity(action_space.__len__(), dtype=int).tolist()

# Model hyperparameters
stack_size = 4

frame_shape = [73, 172]
stacked_frame_shape = [*frame_shape, stack_size]
state_size = stacked_frame_shape      # Our input is a stack of 4 frames hence 84x84x4 (Width, height, channels)
action_size = possible_actions.__len__()
learning_rate = 0.0002      # Alpha (aka learning rate)

# Training hyperparameters
total_episodes = 100        # Total episodes for training
max_steps = 12000            # Max possible steps in an episode
batch_size = 32

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability
decay_rate = 0.00001           # exponential decay rate for exploration prob

# Q learning hyperparameters
gamma = 0.99

# Memory hyperparameters
pretrain_length = batch_size
memory_size = 50000

viewer = SimpleImageViewer()

env = retro.make('TopGear2-Genesis', args.state or retro.STATE_DEFAULT)


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


# high — possible_actions instance
# low — action_space instance
def convert_action(high=None, low=None):
    if high:
        return action_space[possible_actions.index(high)]
    elif low:
        return possible_actions[action_space.index(low)]


stacked_frames = deque([np.zeros(frame_shape, dtype=np.int) for i in range(stack_size)], maxlen=4)

# Reset the graph
tf.reset_default_graph()

# Instantiate the DQNetwork
DQNetwork = DQNetwork(state_size, action_size, learning_rate, batch_size)

# Instantiate memory
memory = Memory(max_size=memory_size)

for i in range(pretrain_length):
    if env.img is not None:
        state = stack_frames(stacked_frames, process_image(env.img))
    else:
        state = np.zeros(stacked_frame_shape, dtype=np.int)

    ac = random.choice(action_space)
    ob, rew, done, info = env.step(ac)

    if done:
        # We finished the episode
        next_state = np.zeros(stacked_frame_shape, dtype=np.int)

        # Add experience to memory
        memory.add((state, convert_action(low=ac), rew, next_state, done))

        # Start a new episode
        env.reset()
    else:
        # Get the next state
        next_state = process_image(ob)
        next_state = stack_frames(stacked_frames, next_state)

        # Add experience to memory
        memory.add((state, convert_action(low=ac), rew, next_state, done))

        # Our state is now the next_state
        state = next_state


# Setup TensorBoard Writer
writer = tf.summary.FileWriter("tensorboard/dqn/1")

## Losses
tf.summary.scalar("Loss", DQNetwork.loss)

write_op = tf.summary.merge_all()

# Saver will help us to save our model
saver = tf.train.Saver()

config = tf.ConfigProto()
config.intra_op_parallelism_threads = 0
config.inter_op_parallelism_threads = 0
with tf.Session(config=config) as sess:
    # Initialize the variables
    # sess.run(tf.global_variables_initializer())
    saver.restore(sess, "./models/model-2018-05-21 20:56:02.ckpt")

    decay_step = 0
    min_position = 20

    for episode in range(total_episodes):
        # Make new episode
        env.reset()
        total_reward = 0
        step = 0

        # Observe the first state
        frame = process_image(env.img)
        state = stack_frames(stacked_frames, frame)

        while step < max_steps:
            step += 1
            # Increase decay_step
            decay_step += 1

            # EPSILON GREEDY STRATEGY
            # Choose action a from state s using epsilon greedy.
            # First we randomize a number
            exp_exp_tradeoff = np.random.rand()

            # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
            explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

            if explore_probability > exp_exp_tradeoff:
                # Make a random action
                action = random.choice(possible_actions)
            else:
                # Get action from Q-network
                # Estimate the Qs values state
                Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})

                # Take the biggest Q value (= the best action)
                action = np.argmax(Qs)
                action = possible_actions[int(action)]
            # Do the action
            prev_state = env.img
            next_state, reward, done, info = env.step(convert_action(high=action))
            while np.array_equal(process_image(prev_state), process_image(next_state)):
                step += 1
                next_state, reward, done, info = env.step(convert_action(high=action))
            total_reward += reward

            render(env.img)

            # If the game is finished
            if done or info.get('lap') == 1:
                # the episode ends so no next state
                next_state = np.zeros(frame_shape, dtype=np.int)
                next_state = stack_frames(stacked_frames, next_state)

                # Set step = max_steps to end the episode
                step = max_steps

                memory.add((state, action, reward, next_state, done))

            else:
                # Get the next state
                next_state = stack_frames(stacked_frames, process_image(next_state))

                # Add experience to memory
                memory.add((state, action, reward, next_state, done))
                state = next_state

            # LEARNING PART
            # Obtain random mini-batch from memory
            batch = memory.sample(batch_size)
            states = np.array([each[0] for each in batch])
            actions = np.array([each[1] for each in batch])
            rewards = np.array([each[2] for each in batch])
            next_states = np.array([each[3] for each in batch])
            dones = np.array([each[4] for each in batch])

            target_Qs_batch = []

            # Get Q values for next_state
            target_Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: next_states})

            # Set Qhat = r if the episode ends at +1, otherwise set Qhat = r + gamma*maxQ(s', a')
            for i in range(0, len(batch)):
                terminal = dones[i]

                # If we are in a terminal state, only equals reward
                if terminal:
                    target_Qs_batch.append(rewards[i])
                else:
                    target = rewards[i] + gamma * np.max(target_Qs[i])
                    target_Qs_batch.append(target)

            targets = np.array([each for each in target_Qs_batch])

            loss, _ = sess.run([DQNetwork.loss, DQNetwork.optimizer],
                               feed_dict={DQNetwork.inputs_: states,
                                          DQNetwork.target_Q: targets,
                                          DQNetwork.actions_: actions})

            # Write TF Summaries
            summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states,
                                                    DQNetwork.target_Q: targets,
                                                    DQNetwork.actions_: actions})
            writer.add_summary(summary, episode)
            writer.flush()

        print('Episode: {}'.format(episode),
              'Total reward: {}'.format(total_reward),
              'Training loss: {:.4f}'.format(loss),
              'Explore P: {:.4f}'.format(explore_probability))

        # Save model every 5 episodes
        if episode % 5 == 0:
            time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            saver.save(sess, "./models/model-{}.ckpt".format(time))
            saver.save(sess, "./models/latest_model.ckpt")
            print("Model Saved")

            subprocess.run(['python', 'play.py'])
