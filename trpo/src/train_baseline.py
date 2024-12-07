import pickle
import gym
import numpy as np
from policy import Policy
from value_function import NNValueFunction
import scipy.signal
from utils import Logger, Scaler
from datetime import datetime
import os
import argparse
import tensorflow as tf


model_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '../model'))


def init_gym(env_name):
    env = gym.make(env_name)
    env.action_space.seed(0)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    return env, obs_dim, act_dim


def run_episode(env, policy, scaler,):
    obs = env.reset()
    observes, actions, rewards, unscaled_obs = [], [], [], []
    done = False
    step = 0.0
    scale, offset = scaler.get()
    scale[-1] = 1.0
    offset[-1] = 0.0
    i = 0
    obs = obs[0]
    while not done and i < 1000:
        obs = obs.astype(np.float64).reshape((1, -1))
        obs = np.append(obs, [[step]], axis=1)
        unscaled_obs.append(obs)
        obs = (obs - offset) * scale
        observes.append(obs)
        action = policy.sample(obs).reshape((1, -1)).astype(np.float64)
        actions.append(action)
        action = action[0]
        obs, reward, done, _, _ = env.step(action)
        if not isinstance(reward, float):
            reward = np.asscalar(reward)
        rewards.append(reward)
        step += 1e-3
        i += 1
    return (np.concatenate(observes), np.concatenate(actions),
            np.array(rewards, dtype=np.float64), np.concatenate(unscaled_obs))


def run_policy(env, policy, scaler, logger, episodes, episode, save_x_episode_model):
    global model_path

    total_steps = 0
    trajectories = []
    for e in range(episodes):
        observes, actions, rewards, unscaled_obs = run_episode(env, policy, scaler)
        total_steps += observes.shape[0]
        trajectory = {'observes': observes,
                      'actions': actions,
                      'rewards': rewards,
                      'unscaled_obs': unscaled_obs}
        trajectories.append(trajectory)
    unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
    scaler.update(unscaled)
    scalar_data = {"vars": scaler.vars, "means": scaler.means, "m": scaler.m}
    episode += 20
    if(episode % save_x_episode_model == 0 and episode != 5 and episode != 0):
        if not os.path.exists(model_path + '/' + str(episode) + '/info'):
            os.makedirs(model_path + '/' + str(episode) + '/info')
        with open(model_path + '/' + str(episode) + "/info/scalar.pkl", "wb") as f:
            pickle.dump(scalar_data, f)

    logger.log({'_MeanReward': np.mean([t['rewards'].sum() for t in trajectories]),
                'Steps': total_steps})

    return trajectories


def discount(x, gamma):
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


def add_disc_sum_rew(trajectories, gamma):
    for trajectory in trajectories:
        if gamma < 0.999:
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        disc_sum_rew = discount(rewards, gamma)
        trajectory['disc_sum_rew'] = disc_sum_rew


def add_value(trajectories, val_func):
    for trajectory in trajectories:
        observes = trajectory['observes']
        values = val_func.predict(observes)
        trajectory['values'] = values


def add_gae(trajectories, gamma, lam):
    for trajectory in trajectories:
        if gamma < 0.999:
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
        values = trajectory['values']

        tds = rewards - values + np.append(values[1:] * gamma, 0)
        advantages = discount(tds, gamma * lam)
        trajectory['advantages'] = advantages


def build_train_set(trajectories):
    observes = np.concatenate([t['observes'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
    advantages = np.concatenate([t['advantages'] for t in trajectories])
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

    return observes, actions, advantages, disc_sum_rew


def log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode):
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


def main(num_episodes, gamma, lam, kl_targ, batch_size, save_x_episode_model, seed, env_name):
    global model_path

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

    env, obs_dim, act_dim = init_gym(env_name)
    obs_dim += 1

    now = datetime.now().strftime("%Y-%m-%d_%H" + 'h' + "_%M" + 'm' + "_%S" + 's' + '--' + model_folder)
    logger = Logger(logname=env_name, now=now)
    episode = 0
    scaler = Scaler(obs_dim)
    val_func = NNValueFunction(obs_dim, model_path, save_x_episode_model, seed)
    policy = Policy(obs_dim, act_dim, kl_targ, batch_size, model_path, save_x_episode_model, seed)

    run_policy(env, policy, scaler, logger, 5, episode, save_x_episode_model)

    while episode < num_episodes:
        trajectories = run_policy(env, policy, scaler, logger, batch_size, episode, save_x_episode_model)
        episode += len(trajectories)
        add_value(trajectories, val_func)
        add_disc_sum_rew(trajectories, gamma)
        add_gae(trajectories, gamma, lam)
        observes, actions, advantages, disc_sum_rew = build_train_set(trajectories)
        log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode)
        policy.update(observes, actions, advantages, logger)
        val_func.fit(observes, disc_sum_rew, logger)
        logger.write(display=True)
    logger.close()
    policy.close_sess()
    val_func.close_sess()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int, help='Number of episodes to run',
                        default=200000)
    parser.add_argument('--gamma', type=float, help='Discount factor', default=0.995)
    parser.add_argument('--lam', type=float, help='Lambda for Generalized Advantage Estimation',
                        default=0.98)
    parser.add_argument('--kl_targ', type=float, help='D_KL target value',
                        default=0.003)
    parser.add_argument('--batch_size', type=int,
                        help='Number of episodes per training batch',
                        default=20)
    parser.add_argument('--save_x_episode_model', type=int, help='Save our model every x episodes', default=None)
    parser.add_argument('--seed', type=int, help='Set seed', default=0)
    parser.add_argument('--env_name', type=str, help='Env Name', default=None)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    main(**vars(args))
