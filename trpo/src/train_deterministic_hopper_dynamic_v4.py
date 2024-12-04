#! /usr/bin/env python3
"""
PPO: Proximal Policy Optimization

Written by Patrick Coady (pat-coady.github.io)

PPO uses a loss function and gradient descent to approximate
Trust Region Policy Optimization (TRPO). See these papers for
details:

TRPO / PPO:
https://arxiv.org/pdf/1502.05477.pdf (Schulman et al., 2016)

Distributed PPO:
https://arxiv.org/abs/1707.02286 (Heess et al., 2017)

Generalized Advantage Estimation:
https://arxiv.org/pdf/1506.02438.pdf

And, also, this GitHub repo which was helpful to me during
implementation:
https://github.com/joschu/modular_rl

This implementation learns policies for continuous environments
in the OpenAI Gym (https://gym.openai.com/). Testing was focused on
the MuJoCo control tasks.
"""
import pickle
import gym
import numpy as np
import random
from policy_dynamic import Policy
from value_function_deterministic import NNValueFunction
import scipy.signal
from utils import Logger, Scaler
from datetime import datetime
import os
import argparse
import tensorflow as tf


all_steps = 0
max_it = 125
ep_it = 160

globTimes = []
model_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '../model'))



def init_gym(env_name, seed):
    """
    Initialize gym environment, return dimension of observation
    and action spaces.

    Args:
        env_name: str environment name (e.g. "Humanoid-v1")

    Returns: 3-tuple
        gym environment (object)
        number of observation dimensions (int)
        number of action dimensions (int)
    """
    env = gym.make(env_name)
    env.action_space.seed(seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    return env, obs_dim, act_dim


def run_episode(env, policy, scaler, animate, max_iteration, seed):
    """ Run single episode with option to animate

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        animate: boolean, True uses env.render() method to animate episode

    Returns: 4-tuple of NumPy arrays
        observes: shape = (episode len, obs_dim)
        actions: shape = (episode len, act_dim)
        rewards: shape = (episode len,)
        unscaled_obs: useful for training scaler, shape = (episode len, obs_dim)
    """
    global all_steps
    global max_it
    obs = env.reset()
    observes, actions, rewards, unscaled_obs = [], [], [], []
    done = False
    step = 0.0
    scale, offset = scaler.get()
    scale[-1] = 1.0  # don't scale time step feature
    offset[-1] = 0.0  # don't offset time step feature
    i = 0
    counter = 0
    obs = obs[0]
    while env.is_healthy and i < max_it:
        if animate:
            env.render()
        obs = obs.astype(np.float32).reshape((1, -1))
        obs = np.append(obs, [[step]], axis=1)  # add time step feature
        unscaled_obs.append(obs)
        obs = (obs - offset) * scale  # center and scale observations
        observes.append(obs)
        action = policy.sample(obs).reshape((1, -1)).astype(np.float32)
        actions.append(action)
        action = action[0]
        plus_cost = env.control_cost(action)
        obs, reward, done, _, _ = env.step(action)
        all_steps += 1
        if not isinstance(reward, float):
            reward = np.asscalar(reward)
        rewards.append(reward)
        step += 1e-3  # increment time step feature
        i += 1
        if i == max_it:
            counter = 1

    return (np.concatenate(observes), np.concatenate(actions),
            np.array(rewards, dtype=np.float32), np.concatenate(unscaled_obs), counter)


def run_policy(env, policy, scaler, logger, animate, episode, max_iteration, seed):
    global model_path
    global all_steps
    global ep_it
    global max_it
    """ Run policy and collect data for a minimum of min_steps and min_episodes

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        logger: logger object, used to save stats from episodes
        episodes: total episodes to run

    Returns: list of trajectory dictionaries, list length = number of episodes
        'observes' : NumPy array of states from episode
        'actions' : NumPy array of actions from episode
        'rewards' : NumPy array of (un-discounted) rewards from episode
        'unscaled_obs' : NumPy array of (un-discounted) rewards from episode
    """

    total_steps = 0
    trajectories = []
    all_counter = 0
    for e in range(ep_it):
        observes, actions, rewards, unscaled_obs, counter = run_episode(env, policy, scaler, animate, max_iteration, seed)
        total_steps += observes.shape[0]
        trajectory = {'observes': observes,
                      'actions': actions,
                      'rewards': rewards,
                      'unscaled_obs': unscaled_obs}
        trajectories.append(trajectory)
        all_counter += counter
    print(max_it)

    if max_it == 2000 and all_counter > 7:
        max_it = 4000
        ep_it = 5
    if max_it == 1000 and all_counter > 13:
        max_it = 2000
        ep_it = 10
    if max_it == 500 and all_counter > 27:
        max_it = 1000
        ep_it = 20
    if max_it == 250 and all_counter > 53:
        max_it = 500
        ep_it = 40
    if max_it == 125 and all_counter > 107:
        max_it = 250
        ep_it = 80

    unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
    scaler.update(unscaled)  # update running statistics for scaling observations
    # Save scalar datas
    scalar_data = {"vars": scaler.vars, "means": scaler.means, "m": scaler.m}
    episode += ep_it
    if(policy.all_steps_remainder < all_steps//1150000):
        if not os.path.exists(model_path + '/' + str(all_steps) + '/info'):
            os.makedirs(model_path + '/' + str(all_steps) + '/info')
        with open(model_path + '/' + str(all_steps) + "/info/scalar.pkl", "wb") as f:
            pickle.dump(scalar_data, f)

    logger.log({'_MeanReward': np.mean([t['rewards'].sum() for t in trajectories]),
                'Steps': total_steps})

    print("ALL STEPS: " + str(all_steps))

    return trajectories


def discount(x, gamma):
    """ Calculate discounted forward sum of a sequence at each point """
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


def add_disc_sum_rew(trajectories, gamma):
    """ Adds discounted sum of rewards to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        gamma: discount

    Returns:
        None (mutates trajectories dictionary to add 'disc_sum_rew')
    """
    for trajectory in trajectories:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        disc_sum_rew = discount(rewards, gamma)
        trajectory['disc_sum_rew'] = disc_sum_rew


def add_value(trajectories, val_func):
    """ Adds estimated value to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        val_func: object with predict() method, takes observations
            and returns predicted state value

    Returns:
        None (mutates trajectories dictionary to add 'values')
    """
    for trajectory in trajectories:
        observes = trajectory['observes']
        values = val_func.predict(observes)
        trajectory['values'] = values


def add_gae(trajectories, gamma, lam):
    """ Add generalized advantage estimator.
    https://arxiv.org/pdf/1506.02438.pdf

    Args:
        trajectories: as returned by run_policy(), must include 'values'
            key from add_value().
        gamma: reward discount
        lam: lambda (see paper).
            lam=0 : use TD residuals
            lam=1 : A =  Sum Discounted Rewards - V_hat(s)

    Returns:
        None (mutates trajectories dictionary to add 'advantages')
    """
    for trajectory in trajectories:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        values = trajectory['values']
        # temporal differences
        tds = rewards - values + np.append(values[1:] * gamma, 0)
        advantages = discount(tds, gamma * lam)
        trajectory['advantages'] = advantages


def build_train_set(trajectories):
    """

    Args:
        trajectories: trajectories after processing by add_disc_sum_rew(),
            add_value(), and add_gae()

    Returns: 4-tuple of NumPy arrays
        observes: shape = (N, obs_dim)
        actions: shape = (N, act_dim)
        advantages: shape = (N,)
        disc_sum_rew: shape = (N,)
    """
    observes = np.concatenate([t['observes'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
    advantages = np.concatenate([t['advantages'] for t in trajectories])
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

    return observes, actions, advantages, disc_sum_rew


def log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode):
    """ Log various batch statistics """
    logger.log({'_mean_obs': np.mean(observes),
                '_min_obs': np.min(observes),
                '_max_obs': np.max(observes),
                '_std_obs': np.mean(np.var(observes, axis=0)),
                '_mean_act': np.mean(actions),
                '_min_act': np.min(actions),
                '_max_act': np.max(actions),
                '_std_act': np.mean(np.var(actions, axis=0)),
                '_mean_adv': np.mean(advantages),
                '_min_adv': np.min(advantages),
                '_max_adv': np.max(advantages),
                '_std_adv': np.var(advantages),
                '_mean_discrew': np.mean(disc_sum_rew),
                '_min_discrew': np.min(disc_sum_rew),
                '_max_discrew': np.max(disc_sum_rew),
                '_std_discrew': np.var(disc_sum_rew),
                '_Episode': episode
                })


def main(num_episodes, gamma, lam, kl_targ, batch_size, animate, model_folder, max_iteration, save_x_episode_model, seed, env_name):
    global model_path
    global all_steps
    """ Main training loop

    Args:
        env_name: OpenAI Gym environment name, e.g. 'Hopper-v1'
        num_episodes: maximum number of episodes to run
        gamma: reward discount factor (float)
        lam: lambda from Generalized Advantage Estimate
        kl_targ: D_KL target for policy update [D_KL(pi_old || pi_new)
        batch_size: number of episodes per policy training batch
    """

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

    if save_x_episode_model == None:
        save_x_episode_model = num_episodes

    if model_folder == None:
        model_dirs = os.listdir(model_path)
        model_dirs.sort()
        dir_number = "{:03d}".format(int(model_dirs[-1]) + 1)
        model_folder = str(dir_number)
        model_path = model_path + '/' + str(dir_number)
        os.makedirs(model_path)
    else:
        model_path = model_path + '/' + model_folder

    env, obs_dim, act_dim = init_gym(env_name, seed)
    obs_dim += 1  # add 1 to obs dimension for time step feature (see run_episode())

    now = datetime.now().strftime("%Y-%m-%d_%H" + 'h' + "_%M" + 'm' + "_%S" + 's' + '--' + model_folder)
    logger = Logger(logname=env_name, now=now)
    if(os.path.exists(model_path + '/info/episodes.txt')):
        with open(model_path + '/info/episodes.txt') as f:
            episode = int(f.readlines()[0])
            scaler = Scaler(obs_dim, model_path, True)
            val_func = NNValueFunction(obs_dim, model_path, save_x_episode_model, seed, True)
    else:
        episode = 0
        scaler = Scaler(obs_dim)
        val_func = NNValueFunction(obs_dim, model_path, save_x_episode_model, seed)
    policy = Policy(obs_dim, act_dim, kl_targ, batch_size, model_path, save_x_episode_model, seed, 1150000)
    # run a few episodes of untrained policy to initialize scaler:
    run_policy(env, policy, scaler, logger, animate, episode, max_iteration, seed)
    while all_steps < 23000000:
        trajectories = run_policy(env, policy, scaler, logger, animate, episode, max_iteration, seed)
        episode += len(trajectories)
        add_value(trajectories, val_func)  # add estimated values to episodes
        add_disc_sum_rew(trajectories, gamma)  # calculated discounted sum of Rs
        add_gae(trajectories, gamma, lam)  # calculate advantage
        # concatenate all episodes into single NumPy arrays
        observes, actions, advantages, disc_sum_rew = build_train_set(trajectories)
        # add various stats to training log:
        log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode)
        policy.update(observes, actions, advantages, logger, all_steps)
        val_func.fit(observes, disc_sum_rew, logger, episode)
        logger.write(display=True)
        if all_steps > 23000000:
            policy.save_policy(all_steps)
    logger.close()
    policy.close_sess()
    val_func.close_sess()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment '
                                                  'using Proximal Policy Optimizer'))
    parser.add_argument('-n', '--num_episodes', type=int, help='Number of episodes to run',
                        default=200000)
    parser.add_argument('-g', '--gamma', type=float, help='Discount factor', default=0.995)
    parser.add_argument('-l', '--lam', type=float, help='Lambda for Generalized Advantage Estimation',
                        default=0.98)
    parser.add_argument('-k', '--kl_targ', type=float, help='D_KL target value',
                        default=0.003)
    parser.add_argument('-b', '--batch_size', type=int,
                        help='Number of episodes per training batch',
                        default=20)
    parser.add_argument('-a', '--animate', type=bool,
                        help='Render the animate of humanoid train',
                        default=False)
    parser.add_argument('-mf', '--model_folder', type=str, help='Continue a train from model folder', default=None)
    parser.add_argument('-mi', '--max_iteration', type=int, help='Set max iteration number', default=1000)
    parser.add_argument('-sxem', '--save_x_episode_model', type=int, help='Save our model every x episodes', default=None)
    parser.add_argument('-s', '--seed', type=int, help='Set seed', default=0)
    parser.add_argument('-en', '--env_name', type=str, help='Environment name', default="Hopper-v4")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    physical_devices = tf.config.list_physical_devices('GPU')
    print(physical_devices)
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    main(**vars(args))

