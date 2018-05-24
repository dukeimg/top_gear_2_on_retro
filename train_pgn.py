#!/usr/bin/env python

import argparse
import retro
import numpy as np
import tensorflow as tf

from collections import deque

from pgn.model import PGNetwork

parser = argparse.ArgumentParser()
parser.add_argument('state', nargs='?', help='the initial state file to load, minus the extension')
parser.add_argument('--scenario', '-s', default='scenario', help='the scenario file to load, minus the extension')
parser.add_argument('--verbose', '-v', action='count', default=1, help='increase verbosity (can be specified multiple times)')
parser.add_argument('--quiet', '-q', action='count', default=0, help='decrease verbosity (can be specified multiple times)')
args = parser.parse_args()
verbosity = args.verbose - args.quiet

if verbosity > 0:
    from gym.envs.classic_control.rendering import SimpleImageViewer

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
total_episodes = 1000      # Total episodes for training
max_steps = 3000           # Max possible steps in an episode
batch_size = 3000

# Exploration parameters for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability
decay_rate = 0.00001           # exponential decay rate for exploration prob

# Q learning hyperparameters
gamma = 0.99

if verbosity > 0:
    viewer = SimpleImageViewer()

env = retro.make('TopGear2-Genesis', state=retro.STATE_DEFAULT, scenario=args.scenario)


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


def discount_and_normalize_rewards(episode_rewards):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamma + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative

    mean = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards) or 1
    discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)

    return discounted_episode_rewards


stacked_frames = deque([np.zeros(frame_shape, dtype=np.int) for i in range(stack_size)], maxlen=4)

# Reset the graph
tf.reset_default_graph()

# Instantiate the PGNetwork
PGNetwork = PGNetwork(state_size, action_size, learning_rate)

# Saver
saver = tf.train.Saver()

# Initialize Session
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
# saver.restore(sess, "./models/model2.ckpt")

# Setup TensorBoard Writer# Setup
writer = tf.summary.FileWriter("tensorboard/pg/test")

## Losses
tf.summary.scalar("Loss", PGNetwork.loss)

## Reward mean
tf.summary.scalar("Reward_mean", PGNetwork.mean_reward_)

write_op = tf.summary.merge_all()


def make_batch(batch_size):
    # Initialize lists: states, actions, rewards, discountedRewards
    states, actions, rewards, rewardsFeed, discountedRewards = [], [], [], [], []

    # Keep track of how many episodes in our batch (useful when we'll need to calculate the average reward per episode)
    episode_num = 1

    # Launch a new episode
    env.reset()

    # Get a new state
    frame = process_image(env.img)
    state = stack_frames(stacked_frames, frame)

    step = 0
    while True:
        step += 1
        # Run State Through Policy & Calculate Action
        action_probability_distribution = sess.run(PGNetwork.action_distribution,
                                                   feed_dict={
                                                       PGNetwork.inputs_: state.reshape(1, *frame_shape, stack_size)
                                                   })
        action = np.random.choice(range(action_probability_distribution.shape[1]),
                                  p=action_probability_distribution.ravel())  # select action w.r.t the actions prob
        action = possible_actions[action]

        # Perform action
        prev_state = env.img
        next_state, reward, done, info = env.step(convert_action(high=action))
        while np.array_equal(process_image(prev_state), process_image(next_state)):
            next_state, reward, done, info = env.step(convert_action(high=action))

        if verbosity > 0:
            render(env.img)

        # Store results
        states.append(state)
        rewards.append(reward)

        # For actions because we output only one (the index) we need (None, 3) (1 is for the action taken)
        # action_ = np.zeros((action_size, action_size))
        # action_[action][action] = 1

        actions.append(action)

        if done or step == max_steps or info.get('lap') == 1:
            step = 0
            # the episode ends so no next state
            rewardsFeed.append(rewards)

            # Calculate gamma Gt
            discountedRewards.append(discount_and_normalize_rewards(rewards))

            if len(np.concatenate(rewardsFeed)) > batch_size:
                break

            # Reset the transition stores
            rewards = []

            # Add episode
            episode_num += 1

            # New episode
            env.reset()

        # If not done, the new_state become the current state
        new_state = process_image(env.img)
        state = stack_frames(stacked_frames, new_state)

    return np.stack(np.array(states)), np.stack(np.array(actions)), np.concatenate(rewardsFeed), np.concatenate(
        discountedRewards), episode_num


allRewards = []
total_rewards = 0
maximumRewardRecorded = 0
mean_reward_total = []
epoch = 1
average_reward = []
while epoch < total_episodes + 1:
    # Gather training data

    batch_states, batch_actions, batch_rewards, batch_discounted_rewards, batch_number_of_episodes =\
        make_batch(batch_size)

    # Calculate the total reward ot the batch
    total_reward_of_that_batch = np.sum(batch_rewards)
    allRewards.append(total_reward_of_that_batch)

    # Calculate the mean reward of the batch
    mean_reward_of_that_batch = np.divide(total_reward_of_that_batch, batch_number_of_episodes)
    mean_reward_total.append(mean_reward_of_that_batch)
    mean_reward = np.divide(np.sum(mean_reward_total), epoch)

    # Calculate the average reward of all training
    # mean_reward_of_that_batch / epoch
    average_reward_of_all_training = np.divide(np.sum(mean_reward_total), epoch)

    # Calculate maximum reward recorded
    maximumRewardRecorded = np.amax(allRewards)

    print("==========================================")
    print("Epoch: ", epoch, "/", total_episodes)
    print("-----------")
    print("Number of training episodes: {}".format(batch_number_of_episodes))
    print("Total reward: {}".format(total_reward_of_that_batch))
    print("Mean Reward of that batch {}".format(mean_reward_of_that_batch))
    print("Average Reward of all training: {}".format(average_reward_of_all_training))
    print("Max reward for a batch so far: {}".format(maximumRewardRecorded))

    # Feedforward, gradient and backpropagation
    loss_, _ = sess.run([PGNetwork.loss, PGNetwork.train_opt],
                        feed_dict={PGNetwork.inputs_: batch_states.reshape((len(batch_states), *frame_shape, stack_size)),
                                   PGNetwork.actions: batch_actions,
                                   PGNetwork.discounted_episode_rewards_: batch_discounted_rewards
                                   })

    # Write TF Summaries
    summary = sess.run(write_op,
                       feed_dict={PGNetwork.inputs_: batch_states.reshape((len(batch_states), *frame_shape, stack_size)),
                                  PGNetwork.actions: batch_actions,
                                  PGNetwork.discounted_episode_rewards_: batch_discounted_rewards,
                                  PGNetwork.mean_reward_: mean_reward
                                  })

    writer.add_summary(summary, epoch)
    writer.flush()

    # Save Model
    if epoch % 10 == 0:
        saver.save(sess, "./models/model-{}.ckpt".format(epoch))
        print("Model saved")
    epoch += 1
